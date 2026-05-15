-- Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).
SELECT city,
    AVG(temp) as avg_temp
FROM hbtn_0c_0 ORDER BY avg_temp DESC;