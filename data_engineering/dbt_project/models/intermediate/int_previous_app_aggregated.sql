SELECT
    sk_id_curr,
    COUNT(*) AS prev_app_count,
    SUM(CASE WHEN name_contract_status = 'Approved' THEN 1 ELSE 0 END) AS prev_approved,
    SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END) AS prev_refused,
    AVG(amt_application) AS avg_prev_application_amount,
    AVG(amt_credit) AS avg_prev_credit_amount,
    AVG(CASE WHEN amt_application > 0
         THEN amt_credit / amt_application ELSE NULL END) AS avg_credit_to_application_ratio,
    MAX(days_decision) AS most_recent_prev_app_days
FROM {{ source('raw', 'previous_application') }}
GROUP BY sk_id_curr
