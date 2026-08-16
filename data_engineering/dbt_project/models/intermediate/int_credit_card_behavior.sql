SELECT
    sk_id_curr,
    AVG(amt_balance / NULLIF(amt_credit_limit_actual, 0)) AS avg_cc_utilization,
    MAX(amt_balance / NULLIF(amt_credit_limit_actual, 0)) AS max_cc_utilization,
    AVG(sk_dpd) AS avg_days_past_due_cc,
    MAX(sk_dpd) AS max_days_past_due_cc,
    COUNT(DISTINCT sk_id_prev) AS num_credit_cards,
    SUM(amt_drawings_atm_current) AS total_atm_drawings,
    AVG(cnt_drawings_current) AS avg_monthly_drawings
FROM {{ source('raw', 'credit_card_balance') }}
GROUP BY sk_id_curr
