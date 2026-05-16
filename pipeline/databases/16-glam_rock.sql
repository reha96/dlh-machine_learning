-- Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity
SELECT band_name, (2020 - formed) AS life_span
FROM metal_bands
WHERE
    split IS NULL AND style LIKE 'Glam rock'
ORDER BY life_span DESC