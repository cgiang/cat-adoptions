-- Median LOS and count among adoptions by age group at intake
WITH ranked as (
	SELECT 
		age_group_intake, 
		length_of_stay_days,
		COUNT(*) OVER(PARTITION BY age_group_intake) cnt,
		ROW_NUMBER() OVER(
			PARTITION BY age_group_intake 
            ORDER BY length_of_stay_days, age_group_intake
		) row_num
	FROM aac_processed
    WHERE is_adopted = 1
)
SELECT 
	age_group_intake, 
	cnt adoption_count, 
    ROUND(AVG(length_of_stay_days), 0) median_los_days
FROM ranked
WHERE row_num IN (FLOOR((cnt+1)/2), CEIL((cnt+1)/2))
GROUP BY age_group_intake
ORDER BY adoption_count DESC;