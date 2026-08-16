WITH raw AS (
    SELECT * FROM {{ source('raw', 'application_train') }}
)
SELECT
    sk_id_curr,
    target,
    name_contract_type,
    code_gender,
    CASE WHEN flag_own_car = 'Y' THEN 1 ELSE 0 END AS owns_car,
    CASE WHEN flag_own_realty = 'Y' THEN 1 ELSE 0 END AS owns_realty,
    amt_income_total,
    amt_credit,
    amt_annuity,
    amt_goods_price,
    ABS(days_birth) / 365.25 AS age_years,
    CASE
        WHEN days_employed = 365243 THEN NULL  -- sentinel value for unemployed/retired
        ELSE ABS(days_employed) / 365.25
    END AS years_employed,
    cnt_children,
    name_education_type,
    name_family_status,
    occupation_type,
    organization_type
FROM raw
