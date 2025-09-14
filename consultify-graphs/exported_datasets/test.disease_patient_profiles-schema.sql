/*!40014 SET FOREIGN_KEY_CHECKS=0*/;
/*!40101 SET NAMES binary*/;
CREATE TABLE `disease_patient_profiles` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `disease` varchar(255) NOT NULL,
  `fever` enum('Yes','No') NOT NULL,
  `cough` enum('Yes','No') NOT NULL,
  `fatigue` enum('Yes','No') NOT NULL,
  `difficulty_breathing` enum('Yes','No') NOT NULL,
  `age` int(11) NOT NULL,
  `gender` enum('Male','Female') NOT NULL,
  `blood_pressure` enum('Low','Normal','High') NOT NULL,
  `cholesterol_level` enum('Low','Normal','High') NOT NULL,
  `outcome_variable` enum('Positive','Negative') NOT NULL,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) /*T![clustered_index] CLUSTERED */
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin AUTO_INCREMENT=30001;
