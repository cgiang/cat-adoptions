-- Monthly change in intake volume and adoption rate 
WITH stats as (
	SELECT 
		DATE_FORMAT(datetime_intake, '%Y-%m') intake_month,
		COUNT(*) intake_count, 
		AVG(is_adopted) * 100 adoption_rate_pct
	FROM aac_processed
    WHERE has_outcome = 1
	GROUP BY intake_month
	ORDER BY intake_month)
SELECT 
	intake_month,
    intake_count,
    ROUND(adoption_rate_pct, 2) adoption_rate_pct,
    ROUND((intake_count / (LAG(intake_count, 1) OVER()) - 1)*100, 2) change_in_intakes_pct,
    ROUND((adoption_rate_pct - LAG(adoption_rate_pct, 1) OVER()), 2) change_in_adoption_rate
 FROM stats;