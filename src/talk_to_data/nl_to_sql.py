import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from groq import Groq
from src.utils.config import GROQ_API_KEY, GROQ_MODEL
from src.utils.helpers import get_db_schema, run_sql_query
from src.talk_to_data.prompt_templates import get_system_prompt, get_answer_prompt
from src.utils.logger import get_logger

logger = get_logger("nl_to_sql")


def get_client():
    """Lazy client initialization — avoids import-time proxy errors."""
    return Groq(api_key=GROQ_API_KEY)


def nl_to_sql(question: str, schema: str = None) -> str:
    """Convert natural language question to SQL using Groq LLM."""
    if schema is None:
        schema = get_db_schema()

    system_prompt = get_system_prompt(schema)

    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    sql = response.choices[0].message.content.strip()

    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()

    logger.info(f"Generated SQL: {sql[:100]}...")
    return sql


def generate_answer(question: str, sql: str, df) -> str:
    """Use Groq to turn SQL results into a plain English business insight."""
    if df is None or len(df) == 0:
        return "No results found for this query."

    results_str = df.head(10).to_string(index=False)
    prompt = get_answer_prompt(question, sql, results_str)

    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def ask(question: str) -> dict:
    """
    Full pipeline: question → SQL → run → plain English answer.
    Returns dict with sql, dataframe, answer, and error if any.
    """
    logger.info(f"Question: {question}")

    try:
        schema = get_db_schema()
        sql    = nl_to_sql(question, schema)

        if sql == "CANNOT_ANSWER":
            return {
                "question": question,
                "sql":      None,
                "data":     None,
                "answer":   "I cannot answer this question with the available data.",
                "error":    None,
            }

        df, error = run_sql_query(sql)

        if error:
            logger.warning(f"SQL error: {error}")
            return {
                "question": question,
                "sql":      sql,
                "data":     None,
                "answer":   f"Query failed: {error}",
                "error":    error,
            }

        answer = generate_answer(question, sql, df)
        logger.info(f"Answer generated successfully")

        return {
            "question": question,
            "sql":      sql,
            "data":     df,
            "answer":   answer,
            "error":    None,
        }
    except Exception as e:
        logger.error(f"Error in Talk-to-Data pipeline: {e}")
        return {
            "question": question,
            "sql":      None,
            "data":     None,
            "answer":   f"An error occurred while connecting to the AI model or running the query: {e}. Please check your internet connection and verify that your GROQ_API_KEY in the .env file is correct and active.",
            "error":    str(e),
        }