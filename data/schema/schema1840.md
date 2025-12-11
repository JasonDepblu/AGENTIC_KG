# Target Schema

> Knowledge graph schema designed from raw data

## Node Types

### Feature
- **Unique Property**: `feature_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、针对外观设计中的“车型整体”方面您觉得该车型优劣点的体验是?—优秀点|13、劣势点|20、针对内饰设计及质感中的“内饰设计风格与布局”方面您觉得该车型优劣点的体验是?—优秀点|20、劣势点|21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点|31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点|31、劣势点|59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点|68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点|68、劣势点|77、针对服务体验中的“接待与专业性”方面您觉得该车型优劣点的体验是？—优秀点|77、劣势点`

### Component
- **Unique Property**: `component_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、针对外观设计中的“车型整体”方面您觉得该车型优劣点的体验是?—优秀点|13、劣势点|20、针对内饰设计及质感中的“内饰设计风格与布局”方面您觉得该车型优劣点的体验是?—优秀点|20、劣势点|21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点|31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点|31、劣势点|59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点|68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点|68、劣势点|77、针对服务体验中的“接待与专业性”方面您觉得该车型优劣点的体验是？—优秀点|77、劣势点`

### Experience
- **Unique Property**: `experience_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、针对外观设计中的“车型整体”方面您觉得该车型优劣点的体验是?—优秀点|13、劣势点|20、针对内饰设计及质感中的“内饰设计风格与布局”方面您觉得该车型优劣点的体验是?—优秀点|20、劣势点|21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点|31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点|31、劣势点|59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点|68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点|68、劣势点|77、针对服务体验中的“接待与专业性”方面您觉得该车型优劣点的体验是？—优秀点|77、劣势点`

### Issue
- **Unique Property**: `issue_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、劣势点|20、劣势点|21、劣势点|31、劣势点|59、劣势点|68、劣势点|77、劣势点`

### Respondent
- **Unique Property**: `respondent_id`
- **Properties**: `age`, `gender`, `family_status`, `vehicle_ownership`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`序号`

## Relationship Types

### MENTIONED_FEATURE
- **Pattern**: `(Respondent)-[MENTIONED_FEATURE]->(Feature)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、针对外观设计中的“车型整体”方面您觉得该车型优劣点的体验是?—优秀点|13、劣势点|20、针对内饰设计及质感中的“内饰设计风格与布局”方面您觉得该车型优劣点的体验是?—优秀点|20、劣势点|21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点|31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点|31、劣势点|59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点|68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点|68、劣势点|77、针对服务体验中的“接待与专业性”方面您觉得该车型优劣点的体验是？—优秀点|77、劣势点`

### MENTIONED_COMPONENT
- **Pattern**: `(Respondent)-[MENTIONED_COMPONENT]->(Component)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、劣势点|20、劣势点|21、劣势点|31、劣势点|59、劣势点|68、劣势点|77、劣势点`

### MENTIONED_EXPERIENCE
- **Pattern**: `(Respondent)-[MENTIONED_EXPERIENCE]->(Experience)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、劣势点|20、劣势点|21、劣势点|31、劣势点|59、劣势点|68、劣势点|77、劣势点`

### MENTIONED_ISSUE
- **Pattern**: `(Respondent)-[MENTIONED_ISSUE]->(Issue)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、劣势点|20、劣势点|21、劣势点|31、劣势点|59、劣势点|68、劣势点|77、劣势点`

### HAS_ISSUE
- **Pattern**: `(Feature)-[HAS_ISSUE]->(Issue)`
- **Properties**: `severity`, `evidence_context`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`13、劣势点|20、劣势点|21、劣势点|31、劣势点|59、劣势点|68、劣势点|77、劣势点`

### RELATES_TO_COMPONENT
- **Pattern**: `(Feature)-[RELATES_TO_COMPONENT]->(Component)`
- **Properties**: `position`, `material`, `visual_property`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`20、针对内饰设计及质感中的“内饰设计风格与布局”方面您觉得该车型优劣点的体验是?—优秀点|20、劣势点|21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点|59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点|68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点|68、劣势点`

### HAS_EXPERIENCE_WITH
- **Pattern**: `(Respondent)-[HAS_EXPERIENCE_WITH]->(Experience)`
- **Properties**: `sentiment`, `modality`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点|21、劣势点`

### INFLUENCES_EXPERIENCE
- **Pattern**: `(Feature)-[INFLUENCES_EXPERIENCE]->(Experience)`
- **Properties**: `valence`, `intensity`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`21、劣势点`

### EVALUATES_AS_POSITIVE
- **Pattern**: `(Experience)-[EVALUATES_AS_POSITIVE]->(Feature)`
- **Properties**: `sentiment_score`, `evidence_text`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`21、针对内饰设计及质感中的“用料及工艺”方面您觉得该车型优劣点的体验是?—优秀点`

### COMPOSED_OF
- **Pattern**: `(Feature)-[COMPOSED_OF]->(Component)`
- **Properties**: `proportion`, `material_type`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`21、劣势点`

### EXPERIENCED_ISSUE
- **Pattern**: `(Experience)-[EXPERIENCED_ISSUE]->(Issue)`
- **Properties**: `severity`, `frequency`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`31、劣势点`

### HAS_EXPERIENCE_OF
- **Pattern**: `(Respondent)-[HAS_EXPERIENCE_OF]->(Experience)`
- **Properties**: `sentiment`, `modality`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点`

### RELATES_TO
- **Pattern**: `(Experience)-[RELATES_TO]->(Feature)`
- **Properties**: `aspect_role`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`31、针对针对智能座舱中的“语音助手”方面您觉得该车型优劣点的体验是?—优秀点`

### HAS_COMPONENT
- **Pattern**: `(Feature)-[HAS_COMPONENT]->(Component)`
- **Properties**: `role_in_feature`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点|59、劣势点`

### EXPERIENCED_AS
- **Pattern**: `(Respondent)-[EXPERIENCED_AS]->(Experience)`
- **Properties**: `intensity`, `context`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点`

### RELATES_TO_ISSUE
- **Pattern**: `(Experience)-[RELATES_TO_ISSUE]->(Issue)`
- **Properties**: `mitigation_status`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`59、针对乘坐舒适中的“座椅舒适性”方面您觉得该车型优劣点的体验是？—优秀点`

### ENABLES_EXPERIENCE
- **Pattern**: `(Feature)-[ENABLES_EXPERIENCE]->(Experience)`
- **Properties**: `valence`, `aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`68、针对操作便利中的“储物空间”方面您觉得该车型优劣点的体验是？—优秀点`

### IMPACTS_EXPERIENCE
- **Pattern**: `(Feature)-[IMPACTS_EXPERIENCE]->(Experience)`
- **Properties**: `experience_type`, `valence`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`59、劣势点`

### REPORTED_EXPERIENCE
- **Pattern**: `(Respondent)-[REPORTED_EXPERIENCE]->(Experience)`
- **Properties**: `sentiment`, `context`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`77、劣势点`

### IDENTIFIED_ISSUE
- **Pattern**: `(Respondent)-[IDENTIFIED_ISSUE]->(Issue)`
- **Properties**: `severity`, `evidence_text`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`77、劣势点`
