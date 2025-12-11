# Target Schema

> Knowledge graph schema designed from raw data

## Node Types

### Respondent
- **Unique Property**: `respondent_id`
- **Properties**: `respondent_id`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`序号`

### Brand
- **Unique Property**: `brand_name`
- **Properties**: `brand_name`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`8、您本次调研的品牌是?`

### Model
- **Unique Property**: `model_name`
- **Properties**: `model_name`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`9、您本次调研的车型及配置是?`

### Store
- **Unique Property**: `store_name`
- **Properties**: `store_name`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`10、10 您本次到访的门店是`

### Aspect
- **Unique Property**: `aspect_name`
- **Properties**: `aspect_name`
- **Extraction Hints**: source_type=column_header, column_pattern=`_score$`

### Feature
- **Unique Property**: `feature_id`
- **Properties**: `feature_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?优秀点|.*?劣势点`

### Experience
- **Unique Property**: `experience_id`
- **Properties**: `experience_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?体验是？—优秀点|.*?体验是？—劣势点`

### Issue
- **Unique Property**: `issue_id`
- **Properties**: `issue_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?劣势点`

### Component
- **Unique Property**: `component_id`
- **Properties**: `component_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?优秀点|.*?劣势点`

## Relationship Types

### RATES
- **Pattern**: `(Respondent)-[RATES]->(Aspect)`
- **Properties**: `score`
- **Extraction Hints**: source_type=rating_column, column_pattern=`_score$`

### EVALUATED_BRAND
- **Pattern**: `(Respondent)-[EVALUATED_BRAND]->(Brand)`
- **Extraction Hints**: source_type=entity_reference, column_pattern=`8、您本次调研的品牌是?`

### VISITED_STORE
- **Pattern**: `(Respondent)-[VISITED_STORE]->(Store)`
- **Extraction Hints**: source_type=entity_reference, column_pattern=`10、10 您本次到访的门店是`

### MENTIONED_FEATURE
- **Pattern**: `(Respondent)-[MENTIONED_FEATURE]->(Feature)`
- **Properties**: `sentiment`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?优秀点|.*?劣势点`

### HAS_EXPERIENCE
- **Pattern**: `(Respondent)-[HAS_EXPERIENCE]->(Experience)`
- **Properties**: `sentiment`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?体验是？—优秀点|.*?体验是？—劣势点`

### IDENTIFIED_ISSUE
- **Pattern**: `(Respondent)-[IDENTIFIED_ISSUE]->(Issue)`
- **Properties**: `severity`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?劣势点`

### RELATES_TO_COMPONENT
- **Pattern**: `(Feature)-[RELATES_TO_COMPONENT]->(Component)`
- **Properties**: `granularity`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?优秀点|.*?劣势点`

### HAS_EXPERIENCE_OF
- **Pattern**: `(Respondent)-[HAS_EXPERIENCE_OF]->(Experience)`
- **Properties**: `sentiment`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?体验是？—优秀点|.*?体验是？—劣势点`

### HAS_ISSUE
- **Pattern**: `(Feature)-[HAS_ISSUE]->(Issue)`
- **Properties**: `severity`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`.*?劣势点`

### BELONGS_TO
- **Pattern**: `(Model)-[BELONGS_TO]->(Brand)`
- **Extraction Hints**: source_type=foreign_key
