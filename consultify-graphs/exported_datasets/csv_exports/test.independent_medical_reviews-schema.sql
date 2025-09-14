/*!40014 SET FOREIGN_KEY_CHECKS=0*/;
/*!40101 SET NAMES binary*/;
CREATE TABLE `independent_medical_reviews` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `reference_id` varchar(20) NOT NULL,
  `report_year` int(11) NOT NULL,
  `diagnosis_category` varchar(255) DEFAULT NULL,
  `diagnosis_sub_category` varchar(255) DEFAULT NULL,
  `treatment_category` varchar(255) DEFAULT NULL,
  `treatment_sub_category` varchar(255) DEFAULT NULL,
  `determination` varchar(255) NOT NULL,
  `type` varchar(255) NOT NULL,
  `age_range` varchar(20) NOT NULL,
  `patient_gender` varchar(20) DEFAULT NULL,
  `findings` text DEFAULT NULL,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) /*T![clustered_index] CLUSTERED */,
  UNIQUE KEY `reference_id` (`reference_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin AUTO_INCREMENT=30001;
