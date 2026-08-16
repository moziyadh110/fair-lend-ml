SELECT
    a.sk_id_curr,
    a.target,

    -- Application features
    a.age_years,
    a.years_employed,
    a.amt_income_total,
    a.amt_credit,
    a.amt_annuity,
    a.amt_credit / NULLIF(a.amt_income_total, 0) AS credit_income_ratio,
    a.amt_annuity / NULLIF(a.amt_income_total, 0) AS annuity_income_ratio,
    a.amt_goods_price / NULLIF(a.amt_credit, 0) AS goods_to_credit_ratio,
    a.owns_car,
    a.owns_realty,
    a.cnt_children,
    a.name_education_type,
    a.name_family_status,

    -- Bureau features
    COALESCE(b.bureau_loan_count, 0) AS bureau_loan_count,
    COALESCE(b.active_loans, 0) AS active_loans,
    COALESCE(b.closed_loans, 0) AS closed_loans,
    COALESCE(b.avg_days_overdue, 0) AS avg_days_overdue,
    COALESCE(b.max_days_overdue, 0) AS max_days_overdue,
    COALESCE(b.avg_debt_to_credit_ratio, 0) AS bureau_debt_ratio,
    COALESCE(b.total_bureau_credit, 0) AS total_bureau_credit,

    -- Previous application features
    COALESCE(p.prev_app_count, 0) AS prev_app_count,
    COALESCE(p.prev_approved, 0) AS prev_approved,
    COALESCE(p.prev_refused, 0) AS prev_refused,
    COALESCE(p.avg_credit_to_application_ratio, 0) AS credit_application_ratio,

    -- Installment behavior features
    COALESCE(i.late_payments, 0) AS late_payments,
    COALESCE(i.avg_days_late, 0) AS avg_days_late,
    COALESCE(i.payment_amount_volatility, 0) AS payment_volatility,
    COALESCE(i.total_installments, 1) AS total_installments,

    -- Credit card behavior features
    COALESCE(cc.avg_cc_utilization, 0) AS avg_cc_utilization,
    COALESCE(cc.max_cc_utilization, 0) AS max_cc_utilization,
    COALESCE(cc.avg_days_past_due_cc, 0) AS avg_days_past_due_cc,
    COALESCE(cc.num_credit_cards, 0) AS num_credit_cards,

    -- POS cash behavior features
    COALESCE(pos.pos_late_payments, 0) AS pos_late_payments,
    COALESCE(pos.avg_pos_days_past_due, 0) AS avg_pos_days_past_due,

    -- Derived risk signals
    COALESCE(b.active_loans, 0) * a.amt_credit AS exposure_at_risk,
    COALESCE(i.late_payments, 0) * 1.0 / NULLIF(COALESCE(i.total_installments, 1), 0)
        AS late_payment_rate,
    CASE WHEN COALESCE(cc.max_cc_utilization, 0) > 0.8
         AND a.amt_credit / NULLIF(a.amt_income_total, 0) > 5
         THEN 1 ELSE 0 END AS high_util_x_high_leverage,
    CASE WHEN COALESCE(b.bureau_loan_count, 0) > 5
         AND a.age_years < 30
         THEN 1 ELSE 0 END AS many_loans_x_young,

    -- Fairness-sensitive (kept for audit, excluded from model input)
    a.code_gender AS _protected_gender,
    a.age_years AS _protected_age,
    a.name_family_status AS _protected_family_status

FROM {{ ref('stg_applications') }} a
LEFT JOIN {{ ref('int_bureau_aggregated') }} b ON a.sk_id_curr = b.sk_id_curr
LEFT JOIN {{ ref('int_previous_app_aggregated') }} p ON a.sk_id_curr = p.sk_id_curr
LEFT JOIN {{ ref('int_installments_behavior') }} i ON a.sk_id_curr = i.sk_id_curr
LEFT JOIN {{ ref('int_credit_card_behavior') }} cc ON a.sk_id_curr = cc.sk_id_curr
LEFT JOIN {{ ref('int_pos_cash_behavior') }} pos ON a.sk_id_curr = pos.sk_id_curr
