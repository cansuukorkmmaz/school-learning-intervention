# AI-Powered School Learning & Intervention System

> From data to insight, from insight to intervention.

An educational decision-support system that analyzes student performance at the topic level, identifies learning gaps that may be hidden by overall grades, and recommends targeted interventions.

The system also considers the school's existing class schedule and classroom capacity when suggesting whether a student could attend another class that is currently covering the relevant topic.

## Problem

A student's overall exam score does not always reveal which topics they are struggling with.

For example, a student may have an acceptable overall Mathematics grade while having a significant gap in Trigonometry.

This creates several questions for educators:

- Which students need support?
- Which specific topics are causing difficulty?
- Is the problem specific to one student or shared across a class?
- What type of intervention would be appropriate?
- Is there another class currently teaching the same topic?
- Does that class have available capacity?
- Did the student's performance change after an intervention?

This project explores how data can support these decisions without replacing teacher judgment.

## Solution

The system follows the workflow:

**Detect → Analyze → Recommend → Human Approval → Intervene → Reassess**

It analyzes student-topic performance using interpretable rules and produces recommendations such as:

- Targeted practice
- Small-group support sessions
- Cross-class attendance
- Class-wide difficulty review
- Monitoring when there is not enough evidence

Recommendations remain suggestions and require human review.

The system does not automatically move students between classes or make decisions on behalf of teachers.

## Key Features

### 1. Topic-Level Learning Gap Detection

Instead of relying only on overall grades, the system analyzes performance for individual topics.

The analysis considers:

- Recency-weighted performance
- Student's personal baseline
- Class-level context
- Number of observations
- Performance trends

When there is only one observation for a topic, the system avoids making a strong conclusion and marks the case for monitoring.

### 2. Explainable Recommendations

Each recommendation is based on observable signals.

For example:

**Student → Trigonometry gap → Personal baseline comparison → Recent performance → Recommendation**

This makes the reasoning behind a recommendation visible rather than treating the system as a black box.

### 3. Cross-Class Learning Opportunity Matching

One of the main features of the project is matching students with an existing class that may currently be covering the topic they need help with.

The system checks:

1. Whether the student's grade is compatible
2. Whether another class is currently teaching the topic
3. Whether the student's own class should be excluded
4. Whether the timetable is compatible
5. Whether there are enough days remaining
6. Whether the target class has available capacity

The result can be:

- `SUGGESTED`
- `NO_CAPACITY`
- `NO_MATCH_SCHEDULE`

The system only produces a recommendation. Teacher approval is required before any intervention.

### 4. Intervention Outcome Analysis

The system compares student performance before and after an intervention.

It also constructs a comparison group at evaluation time.

The project reports observed changes rather than claiming that an intervention caused the improvement.

## Architecture

The project is organized into separate stages:

```text
Raw Data
   ↓
Validation
   ↓
Preprocessing
   ↓
Student & Topic Analysis
   ↓
Recommendation Engine
   ↓
Human Approval / Intervention
   ↓
Outcome Analysis
```

The architecture separates:

- Data generation
- Data processing
- Analysis
- Recommendation logic
- Feedback / outcome analysis
- Streamlit presentation layer

This separation makes the recommendation logic independent from the UI.

## Data

The current MVP uses synthetic and anonymized data.

The dataset represents:

- 3 Grade 10 classes
- Approximately 60 students
- 5 Mathematics topics
- Student assessments over an 18-week period
- Class schedules
- Classroom capacity
- Teacher information
- Intervention records

The synthetic dataset contains deliberately planted scenarios such as:

- Topic-specific learning gaps
- Multiple learning gaps
- Class-wide difficulty
- Improving performance
- Declining performance
- Insufficient observations
- Missing assessment observations
- Post-intervention performance changes

These scenarios are used to test whether the system behaves as intended.

## Why Rule-Based Instead of Machine Learning?

The current MVP intentionally uses an interpretable rule-based approach.

There is no real labeled dataset available for training a supervised machine learning model, and educational interventions require explanations that teachers can understand.

Therefore, the first version focuses on:

- Explainability
- Transparent decision logic
- Reproducibility
- Human oversight

With sufficient real-world data, the system could later incorporate machine learning models for tasks such as:

- Predicting students at risk of falling behind
- Estimating intervention effectiveness
- Learning personalized intervention strategies
- Detecting more complex performance patterns

## Human Oversight

The system is designed as a decision-support tool rather than an autonomous decision maker.

Recommendations remain in a suggested state until reviewed.

The system does not:

- Rank teachers
- Automatically move students between classes
- Automatically approve interventions
- Label students permanently based on a single observation

The goal is to provide educators with additional information when making decisions.

## Technology Stack

- Python
- Pandas
- NumPy
- Streamlit
- PyYAML
- Pytest

## Project Structure

```text
school-learning-intervention/
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── ground_truth/
│   └── outputs/
│
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── data_generation/
│   ├── processing/
│   ├── analysis/
│   ├── recommendation/
│   └── feedback/
│
├── tests/
│
├── config.yaml
├── requirements.txt
└── README.md
```

## Running the Project

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
python -m streamlit run app/main.py
```

The application will open locally in your browser.

## Example Workflow

A typical recommendation can be interpreted as:

```text
Student S014
      ↓
Trigonometry performance is significantly below
the student's personal baseline
      ↓
The system detects a specific topic gap
      ↓
Targeted practice is recommended
      ↓
A small-group support session may be suggested
      ↓
If another compatible class is currently
covering Trigonometry, cross-class attendance
may also be suggested
      ↓
Teacher reviews the recommendation
      ↓
Intervention takes place
      ↓
Performance is reassessed
```

## Limitations

This is an MVP built with synthetic data.

The current system does not attempt to establish causal relationships between interventions and student outcomes.

Real-world deployment would require:

- Real anonymized student data
- Strong privacy and access controls
- Validation with educators
- More robust evaluation
- Integration with existing school information systems
- More extensive testing across different subjects and grade levels

## Future Work

Potential extensions include:

- Additional subjects
- More detailed curriculum modeling
- Personalized intervention recommendations
- Machine learning-based risk prediction
- Longitudinal student modeling
- More advanced schedule optimization
- Teacher feedback loops
- Integration with real school systems
- Improved evaluation methodologies

## Development Note

GPT and Claude were used as AI-assisted development tools for architecture exploration, implementation support, debugging, and documentation.

Design decisions, validation of outputs, and final integration were reviewed and adapted during development.

## Status

**MVP — Functional Prototype**

The current version demonstrates the complete decision-support workflow using synthetic data.
