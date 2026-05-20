-- Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.

DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)

BEGIN

    -- declare local variable to store avg score
    DECLARE average_score FLOAT;

    -- select and calculate student average score
    SELECT AVG(score) INTO average_score
    FROM corrections
    WHERE user_id = user_id;

    -- Update the user's average_score, handle NULL case
    UPDATE users
    SET average_score = IFNULL(avg_score, 0)  -- Use 0 if no scores exist
    WHERE id = input_user_id;

END $$

DELIMITER ;