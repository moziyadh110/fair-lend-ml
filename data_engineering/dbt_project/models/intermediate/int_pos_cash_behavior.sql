SELECT
    sk_id_curr,
    COUNT(*) AS pos_cash_count,
    SUM(CASE WHEN sk_dpd > 0 THEN 1 ELSE 0 END) AS pos_late_payments,
    AVG(sk_dpd) AS avg_pos_days_past_due,
    MAX(sk_dpd) AS max_pos_days_past_due,
    SUM(CASE WHEN name_contract_status = 'Completed' THEN 1 ELSE 0 END) AS pos_completed,
    SUM(CASE WHEN name_contract_status = 'Active' THEN 1 ELSE 0 END) AS pos_active
FROM {{ source('raw', 'pos_cash_balance') }}
GROUP BY sk_id_curr
