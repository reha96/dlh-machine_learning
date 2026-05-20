-- Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student
DELIMITER $$

CREATE PROCEDURE AddBonus (
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
-- declare local variable to store project_id
    DECLARE project_id INT;
-- create new project name if it doesn't exist yet
   IF NOT EXISTS (SELECT * FROM projects WHERE
projects.name = p_project_name) THEN:
 INSERT INTO projects (name) VALUES (p_project_name)
   END IF;
-- store in local var project id for selected project name
SELECT id INTO project_id FROM projects WHERE name = p_project_name;
-- update corrections table with user_id, project_id, and score
INSERT INTO corrections (user_id, project_id, score)
    VALUES (p_user_id, project_id, p_score);

   END$$

DELIMITER;