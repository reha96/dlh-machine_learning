-- Write a SQL script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(
    IN p_user_id INT
)

BEGIN

    -- declare local variable to store weighted avg score
    DECLARE v_weighted_average_score FLOAT;

    -- select and calculate student weighted average score
    SELECT AVG(score)*projects.weight INTO v_weighted_average_score
    FROM corrections
    JOIN projects ON corrections.project_id = projects.id
    WHERE user_id = p_user_id;

    -- Update the user's weighted average_score, handle NULL case
    UPDATE users
    SET average_weighted_score = IFNULL(v_weighted_average_score, 0)  -- Use 0 if no scores exist
    WHERE id = p_user_id;

END $$

DELIMITER;