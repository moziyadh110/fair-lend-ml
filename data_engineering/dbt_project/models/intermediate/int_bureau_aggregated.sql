SELECT
    sk_id_curr,
    COUNT(*) AS bureau_loan_count,
    SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END) AS active_loans,
    SUM(CASE WHEN credit_active = 'Closed' THEN 1 ELSE 0 END) AS closed_loans,
    AVG(days_credit) AS avg_days_credit,
    AVG(credit_day_overdue) AS avg_days_overdue,
    MAX(credit_day_overdue) AS max_days_overdue,
    SUM(amt_credit_sum) AS total_bureau_credit,
    SUM(amt_credit_sum_debt) AS total_bureau_debt,
    AVG(amt_credit_sum_debt / NULLIF(amt_credit_sum, 0)) AS avg_debt_to_credit_ratio
FROM {{ source('raw', 'bureau') }}
GROUP BY sk_id_curr
