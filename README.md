# Consultify

A medical consultation system that connects patients with appropriate doctors through AI-powered triaging and prescription assistance.

## Overview

Consultify automates the initial medical consultation process by:
- Collecting patient symptoms through an intelligent clerking system
- Matching patients with appropriate medical specialists
- Providing prescription assistance using medical databases
- Supporting multi-language communication

## Core Components

### Models (`models.py`)
- **Consultation**: Tracks consultation state and patient-doctor assignments
- **Message**: Handles multi-language message processing and translation
- **Prescription**: Manages prescription data with timestamps and patient linking
- **AgentState**: Maintains conversation state for AI processing

### Clerking System (`clerking-system.py`)
Medical intake workflow that:
- Gathers patient symptoms and medical history
- Determines when sufficient information is collected
- Routes patients to appropriate medical specialists
- Provides drug information lookup via TiDB vector store

### Prescription Agent (`prescription-agent.py`)
Specialized system for prescription-related queries:
- Searches British National Formulary database
- Provides drug interaction and dosage information
- Grades document relevance for medical queries

### Prompts (`prompts.py`)
Template library for AI interactions including:
- Query generation for medical database searches
- Specialist selection based on symptoms
- Multi-language translation prompts
- Consultation summary generation

## Technology Stack

- **LangGraph**: Workflow orchestration
- **Mistral AI**: Language model for medical reasoning
- **TiDB Vector Store**: Medical knowledge base storage
- **Pydantic**: Data validation and modeling

## Usage

The system processes patient interactions through state-driven workflows, automatically transitioning between clerking, specialist matching, and prescription assistance based on consultation needs.