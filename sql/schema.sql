-- Credit Risk Platform — SQLite Schema
-- Auto-populated from CSV files via src/utils/helpers.py build_sqlite_db()

-- ── Main application table ─────────────────────────────────
-- Source: application_train.csv
-- Rows: ~307,511 | Primary key: SK_ID_CURR
CREATE TABLE IF NOT EXISTS application_train (
    SK_ID_CURR          INTEGER,   -- Unique loan application ID
    TARGET              INTEGER,   -- 1 = defaulted, 0 = did not default
    NAME_CONTRACT_TYPE  TEXT,      -- Cash loans / Revolving loans
    CODE_GENDER         TEXT,      -- M / F
    AMT_INCOME_TOTAL    REAL,      -- Annual income
    AMT_CREDIT          REAL,      -- Loan amount
    AMT_ANNUITY         REAL,      -- Monthly annuity
    AMT_GOODS_PRICE     REAL,      -- Price of goods for loan
    DAYS_BIRTH          INTEGER,   -- Age in days (negative)
    DAYS_EMPLOYED       INTEGER,   -- Employment duration in days (negative)
    EXT_SOURCE_1        REAL,      -- External credit score 1 (0-1)
    EXT_SOURCE_2        REAL,      -- External credit score 2 (0-1)
    EXT_SOURCE_3        REAL,      -- External credit score 3 (0-1)
    NAME_INCOME_TYPE    TEXT,      -- Income source
    NAME_EDUCATION_TYPE TEXT,      -- Education level
    CNT_FAM_MEMBERS     REAL       -- Number of family members
    -- ... 106 additional columns
);

-- ── Bureau credit history ──────────────────────────────────
-- Source: bureau.csv
-- Rows: ~1,716,428 | Foreign key: SK_ID_CURR → application_train
CREATE TABLE IF NOT EXISTS bureau (
    SK_ID_CURR          INTEGER,   -- Links to application_train
    SK_ID_BUREAU        INTEGER,   -- Unique bureau record ID
    CREDIT_ACTIVE       TEXT,      -- Active / Closed / Sold / Bad debt
    CREDIT_CURRENCY     TEXT,      -- Currency of credit
    DAYS_CREDIT         INTEGER,   -- Days before application credit was applied
    CREDIT_DAY_OVERDUE  INTEGER,   -- Days overdue on credit
    AMT_CREDIT_MAX_OVERDUE REAL,   -- Max overdue amount
    AMT_CREDIT_SUM      REAL,      -- Total credit amount
    AMT_CREDIT_SUM_DEBT REAL,      -- Current debt on credit
    AMT_CREDIT_SUM_OVERDUE REAL,   -- Current overdue amount
    CNT_CREDIT_PROLONG  INTEGER    -- Number of times credit was prolonged
);

-- ── Previous applications ──────────────────────────────────
-- Source: previous_application.csv
-- Rows: ~1,670,214 | Foreign key: SK_ID_CURR → application_train
CREATE TABLE IF NOT EXISTS previous_application (
    SK_ID_CURR          INTEGER,   -- Links to application_train
    SK_ID_PREV          INTEGER,   -- Unique previous application ID
    NAME_CONTRACT_TYPE  TEXT,      -- Type of previous loan
    NAME_CONTRACT_STATUS TEXT,     -- Approved / Refused / Canceled / Unused offer
    AMT_CREDIT          REAL,      -- Previous credit amount
    AMT_ANNUITY         REAL,      -- Previous annuity
    AMT_APPLICATION     REAL,      -- Amount applied for
    AMT_GOODS_PRICE     REAL,      -- Previous goods price
    DAYS_DECISION       INTEGER,   -- Days before current application of decision
    NAME_PAYMENT_TYPE   TEXT       -- Payment method
);