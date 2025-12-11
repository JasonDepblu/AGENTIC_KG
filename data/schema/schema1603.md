# Target Schema

> Knowledge graph schema designed from raw data

## Node Types

### CostControlStrategy
- **Unique Property**: `costcontrolstrategy_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`92、该车型是否有成本控制方面的启发项?`

### Respondent
- **Unique Property**: `respondent_id`
- **Extraction Hints**: source_type=entity_selection, column_pattern=`序号`

### Feature
- **Unique Property**: `feature_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`92、该车型是否有成本控制方面的启发项?`

### SceneInnovation
- **Unique Property**: `sceneinnovation_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`90、该车型/品牌是否有优秀的场景创新、亮点设计等启发项?`

### EmotionalNeed
- **Unique Property**: `emotionalneed_id`
- **Properties**: `name`, `category`, `source_column`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`89、您针对该品牌“品牌用户群 ”方面的体验是?（选填）`

### UserSegment
- **Unique Property**: `usersegment_id`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`89、您针对该品牌“品牌用户群 ”方面的体验是?（选填）`

## Relationship Types

### MENTIONED_COST_CONTROL
- **Pattern**: `(Respondent)-[MENTIONED_COST_CONTROL]->(CostControlStrategy)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`92、该车型是否有成本控制方面的启发项?`

### ENABLES_COST_SAVINGS
- **Pattern**: `(CostControlStrategy)-[ENABLES_COST_SAVINGS]->(Feature)`
- **Properties**: `strategy_type`, `cost_impact`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`92、该车型是否有成本控制方面的启发项?`

### MENTIONED_SCENE_INNOVATION
- **Pattern**: `(Respondent)-[MENTIONED_SCENE_INNOVATION]->(SceneInnovation)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`90、该车型/品牌是否有优秀的场景创新、亮点设计等启发项?`

### ENABLES_SCENE_INNOVATION
- **Pattern**: `(Feature)-[ENABLES_SCENE_INNOVATION]->(SceneInnovation)`
- **Properties**: `enabling_feature`, `innovation_type`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`90、该车型/品牌是否有优秀的场景创新、亮点设计等启发项?`

### EXPERIENCED_EMOTIONAL_NEED
- **Pattern**: `(Respondent)-[EXPERIENCED_EMOTIONAL_NEED]->(EmotionalNeed)`
- **Properties**: `sentiment`, `source_aspect`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`89、您针对该品牌“品牌用户群 ”方面的体验是?（选填）`

### HAS_EMOTIONAL_NEED
- **Pattern**: `(UserSegment)-[HAS_EMOTIONAL_NEED]->(EmotionalNeed)`
- **Properties**: `need_type`, `priority`
- **Extraction Hints**: source_type=text_extraction, column_pattern=`89、您针对该品牌“品牌用户群 ”方面的体验是?（选填）`
