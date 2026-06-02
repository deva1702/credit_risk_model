import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.talk_to_data.nl_to_sql import ask
from src.utils.helpers import build_sqlite_db
from src.talk_to_data.prompt_templates import SAMPLE_QUERIES
from src.utils.logger import get_logger

logger = get_logger("query_runner")


def run_all_sample_queries():
    """Run all 5+ required sample queries and print results."""
    print("\n" + "="*65)
    print("     TALK-TO-DATA — SAMPLE QUERY VALIDATION")
    print("="*65)

    passed = 0
    failed = 0

    for i, question in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n[{i}] Question: {question}")
        print("-" * 55)

        result = ask(question)

        if result["error"]:
            print(f"  ✗ ERROR: {result['error']}")
            failed += 1
        elif result["sql"] is None:
            print(f"  ✗ CANNOT ANSWER")
            failed += 1
        else:
            print(f"  SQL     : {result['sql'][:80]}...")
            if result["data"] is not None:
                print(f"  Rows    : {len(result['data'])}")
                print(f"  Preview : \n{result['data'].head(3).to_string(index=False)}")
            print(f"  Answer  : {result['answer']}")
            passed += 1

    print("\n" + "="*65)
    print(f"  Results: {passed} passed / {failed} failed out of {len(SAMPLE_QUERIES)}")
    print("="*65)


if __name__ == "__main__":
    # Step 1: Build the SQLite DB first
    print("Building SQLite database from CSVs...")
    build_sqlite_db()
    print("DB ready.\n")

    # Step 2: Run all sample queries
    run_all_sample_queries()