-- Adoption rate and count by age group at intake
SELECT 
	age_group_intake, 
	COUNT(*) intake_count, 
    ROUND(AVG(is_adopted) * 100, 2) adoption_rate_pct, 
    SUM(is_adopted) adoption_count
FROM aac_processed
WHERE has_outcome = 1
GROUP BY age_group_intake
ORDER BY intake_count DESC;