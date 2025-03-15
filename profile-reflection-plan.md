# Profile Generation Reflection Pattern Implementation

## Overview
Enhance profile generation with reflection pattern to validate and iteratively improve generated profiles.

## Implementation Details

### 1. New Models in profiles.py

```python
class ProfileValidation(BaseModel):
    """Model for validating generated profiles"""
    is_valid: bool
    reason: str
    suggestions: List[str]
    
    class Config:
        json_schema_extra = {
            "examples": [{
                "is_valid": True,
                "reason": "Profile contains all required elements and is well-structured",
                "suggestions": []
            }]
        }
```

### 2. New Validation Functions

```python
def validate_profile(profile: Dict) -> ProfileValidation:
    """
    Validate a generated profile against defined criteria:
    - Has minimum length
    - Contains required sections
    - Properly formatted markdown
    - Valid citations
    - Reasonable confidence score
    """
    # Implementation details...
```

### 3. Enhanced Profile Generation

```python
def generate_profile_with_reflection(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    max_iterations: int = 3
) -> Dict:
    """
    Generate a profile with reflection and improvement.
    Uses iterative_improve from utils.py.
    """
    # Implementation using iterative_improve...
```

### 4. Profile Generation System Prompt

The system prompt for profile generation should be enhanced to include:
- Explicit formatting requirements
- Required sections
- Citation format rules
- Minimum content guidelines

### 5. Integration Points

1. Update process_and_extract.py to use new reflection-enabled functions
2. Modify existing profile generation to use new validation
3. Add logging for reflection process
4. Update tests to cover validation

## Implementation Steps

1. Add ProfileValidation model
2. Implement validation function
3. Create wrapper function using iterative_improve
4. Update system prompts
5. Integrate with process_and_extract.py
6. Add tests

## Success Criteria

- Profiles consistently meet quality standards
- Invalid profiles are caught and improved
- Performance impact is reasonable
- Existing functionality is preserved

## Future Enhancements

- Add more sophisticated validation rules
- Implement profile-specific improvement suggestions
- Add validation metrics tracking