-- Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.

DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN p_user_id INT
)

BEGIN

    -- declare local variable to store avg score
    DECLARE v_average_score FLOAT;

    -- select and calculate student average score
    SELECT AVG(score) INTO v_average_score
    FROM corrections
    WHERE user_id = p_user_id;

    -- Update the user's average_score, handle NULL case
    UPDATE users
    SET average_score = IFNULL(v_average_score, 0)  -- Use 0 if no scores exist
    WHERE id = p_user_id;

END $$

DELIMITER ;