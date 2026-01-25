-- Adoption rate by name presence at intake
SELECT 
	(CASE 
		WHEN name_intake IS NULL OR name_intake = '' THEN "No"
        ELSE "Yes"
	END) has_name_intake,
    COUNT(*) intake_count,
	ROUND(AVG(is_adopted) * 100, 2) adoption_rate_pct
FROM aac_processed
WHERE has_outcome = 1
GROUP BY has_name_intake
ORDER BY adoption_rate_pct DESC;