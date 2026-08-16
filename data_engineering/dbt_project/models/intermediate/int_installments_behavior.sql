SELECT
    sk_id_curr,
    COUNT(*) AS total_installments,
    SUM(CASE WHEN days_entry_payment > days_instalment THEN 1 ELSE 0 END) AS late_payments,
    AVG(CASE WHEN days_entry_payment > days_instalment
         THEN days_entry_payment - days_instalment ELSE 0 END) AS avg_days_late,
    SUM(amt_payment - amt_instalment) AS total_over_under_payment,
    STDDEV(amt_payment) AS payment_amount_volatility
FROM {{ source('raw', 'installments_payments') }}
GROUP BY sk_id_curr
