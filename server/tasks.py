"""
Task definitions and graders for SQL Review Environment.
Each task has: broken_query, schema, error_description, reference_answer, grader.
"""
import re
import sqlglot
from typing import Tuple

TASKS = {
    "fix_syntax_error": {
        "schema_context": """
Tables:
  orders(order_id INT, customer_id INT, amount DECIMAL, order_date DATE, status VARCHAR)
  customers(customer_id INT, name VARCHAR, email VARCHAR, country VARCHAR)
""",
        "broken_query": """
SELECT customer_id, SUM(amount) as total
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING total > 1000
ORDER BY total DESC
LIMIT 10
""",
        "error_description": (
            "This query fails in standard SQL because HAVING clause cannot reference "
            "a column alias defined in SELECT. Fix it so it runs correctly."
        ),
        "hint": "HAVING must reference the aggregate expression, not its alias.",
        "difficulty": "easy",
    },
    "fix_logic_error": {
        "schema_context": """
Tables:
  employees(emp_id INT, name VARCHAR, dept_id INT, salary DECIMAL, hire_date DATE)
  departments(dept_id INT, dept_name VARCHAR, manager_id INT)
""",
        "broken_query": """
SELECT d.dept_name, AVG(e.salary) as avg_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
WHERE e.hire_date > '2020-01-01'
GROUP BY d.dept_name
ORDER BY avg_salary DESC;
""",
        "error_description": (
            "This query is supposed to show ALL departments with their average salary "
            "for employees hired after 2020 (departments with no such employees should show NULL). "
            "However, the WHERE clause silently converts the LEFT JOIN into an INNER JOIN, "
            "dropping departments that have no matching employees."
        ),
        "hint": "Move the filter condition from WHERE into the JOIN ON clause.",
        "difficulty": "medium",
    },
    "optimize_query": {
        "schema_context": """
Tables:
  events(event_id INT, user_id INT, event_type VARCHAR, created_at TIMESTAMP)
  users(user_id INT, username VARCHAR, signup_date DATE, country VARCHAR)

Indexes: events(user_id), events(created_at), users(user_id)
Table sizes: events ~50M rows, users ~1M rows
""",
        "broken_query": """
SELECT u.username, COUNT(*) as event_count
FROM users u
WHERE u.user_id IN (
    SELECT user_id FROM events
    WHERE event_type = 'purchase'
    AND created_at >= '2024-01-01'
)
GROUP BY u.username
ORDER BY event_count DESC
LIMIT 100;
""",
        "error_description": (
            "This query has two problems: "
            "1) The correlated subquery with IN is extremely slow on 50M rows — "
            "it should be rewritten as a JOIN. "
            "2) The COUNT(*) counts users, not events — the GROUP BY u.username with "
            "COUNT(*) actually counts the number of users (always 1 per user since "
            "username is unique) rather than total purchase events per user. "
            "Rewrite to efficiently count purchase events per user using a JOIN."
        ),
        "hint": "Use JOIN instead of IN-subquery; aggregate on the events side.",
        "difficulty": "hard",
    },
}


def normalize_sql(query: str) -> str:
    """Lowercase and strip whitespace for comparison."""
    return re.sub(r'\s+', ' ', query.strip().lower())


def grade_fix_syntax_error(submitted: str) -> Tuple[float, str]:
    """
    Easy task: HAVING clause must reference SUM(amount) not alias.
    Check for absence of 'having total' and presence of 'having sum(amount)'.
    """
    norm = normalize_sql(submitted)

    # Must not use alias in HAVING
    if 'having total' in norm:
        return 0.1, "Still using alias 'total' in HAVING — this fails in standard SQL."

    # Must have HAVING with the aggregate
    if 'having sum(amount)' in norm or 'having sum( amount )' in norm:
        # Try to parse it
        try:
            sqlglot.parse_one(submitted)
            return 1.0, "Correct! HAVING references the aggregate expression directly."
        except Exception:
            return 0.6, "Logic looks right but SQL has parse errors."

    # Partial credit: removed HAVING alias but wrong aggregate reference
    if 'having' in norm:
        return 0.4, "HAVING clause present but not referencing SUM(amount) correctly."

    # Removed HAVING entirely — acceptable alternative using subquery
    if 'select' in norm and 'sum(amount)' in norm:
        return 0.8, "Alternative approach detected (subquery/CTE). Mostly correct."

    return 0.0, "Query does not appear to address the HAVING alias issue."


def grade_fix_logic_error(submitted: str) -> Tuple[float, str]:
    """
    Medium task: WHERE clause filter on LEFT JOIN must move to ON clause.
    """
    norm = normalize_sql(submitted)

    # Must keep LEFT JOIN
    if 'left join' not in norm and 'left outer join' not in norm:
        return 0.2, "Lost the LEFT JOIN — departments with no qualifying employees will be dropped."

    # Filter must be in ON clause, not WHERE
    filter_in_on = bool(re.search(r'on\s+.*hire_date', norm, re.DOTALL))
    filter_in_where = bool(re.search(r'where\s+.*hire_date', norm, re.DOTALL))

    if filter_in_on and not filter_in_where:
        try:
            sqlglot.parse_one(submitted)
            return 1.0, "Correct! Filter moved to ON clause preserves the LEFT JOIN semantics."
        except Exception:
            return 0.7, "Semantically correct but has parse issues."

    if filter_in_on and filter_in_where:
        return 0.5, "Filter is in ON clause but also still in WHERE — removes NULL rows."

    if not filter_in_on and not filter_in_where:
        # Removed the filter entirely
        return 0.3, "Filter removed entirely — incorrect, query should filter by hire_date."

    return 0.2, "Filter still in WHERE clause — LEFT JOIN behaves like INNER JOIN."


def grade_optimize_query(submitted: str) -> Tuple[float, str]:
    """
    Hard task: Replace IN-subquery with JOIN; count events not users.
    Partial credit for each fix.
    """
    norm = normalize_sql(submitted)
    score = 0.0
    feedback_parts = []

    # Check 1: No IN (subquery) pattern
    has_in_subquery = bool(re.search(r'\bin\s*\(select', norm))
    if not has_in_subquery:
        score += 0.35
        feedback_parts.append("✓ No IN-subquery (good for performance).")
    else:
        feedback_parts.append("✗ Still using IN (subquery) — slow on large tables.")

    # Check 2: Uses JOIN
    has_join = 'join' in norm
    if has_join:
        score += 0.25
        feedback_parts.append("✓ Uses JOIN.")
    else:
        feedback_parts.append("✗ No JOIN found.")

    # Check 3: Counts events (aggregate from events table)
    # Should count events — look for count on events side or count(e.event_id) etc.
    counts_events = bool(
        re.search(r'count\s*\(\s*e\.event_id\s*\)', norm) or
        re.search(r'count\s*\(\s*event_id\s*\)', norm) or
        re.search(r'count\s*\(\s*e\.\*\s*\)', norm)
    )
    if counts_events:
        score += 0.25
        feedback_parts.append("✓ Counts events correctly.")
    else:
        # Give partial if they at least have a count
        if 'count(' in norm:
            score += 0.1
            feedback_parts.append("~ Has COUNT but may not be counting events correctly.")
        else:
            feedback_parts.append("✗ No COUNT found.")

    # Check 4: Parses cleanly
    try:
        sqlglot.parse_one(submitted)
        score += 0.15
        feedback_parts.append("✓ Valid SQL syntax.")
    except Exception:
        feedback_parts.append("✗ SQL has syntax errors.")

    score = min(score, 1.0)
    return round(score, 2), " | ".join(feedback_parts)


GRADERS = {
    "fix_syntax_error": grade_fix_syntax_error,
    "fix_logic_error": grade_fix_logic_error,
    "optimize_query": grade_optimize_query,
}