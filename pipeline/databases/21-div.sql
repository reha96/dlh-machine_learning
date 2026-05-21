-- Write a SQL script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.

DELIMITER $$

CREATE FUNCTION SafeDiv(first_num INT, second_num INT)
RETURNS INT
DETERMINISTIC
NO SQL
BEGIN
    IF second_num = 0 THEN 
        RETURN 0;
    ELSE 
        RETURN first_num/second_num;
    END IF;
END $$

DELIMITER;