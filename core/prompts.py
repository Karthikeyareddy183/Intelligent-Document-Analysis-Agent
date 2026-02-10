"""Prompt templates for LLM interactions."""

DOCUMENT_QA_PROMPT = """You are an expert document analyst. Answer the question based ONLY on the provided context. If the context contains tables or chart descriptions, reference them explicitly.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [Page X, Section Y] format
3. If data comes from a table or chart, mention this explicitly
4. If you cannot answer from the context, say "Insufficient context to answer this question."
5. Be concise but complete
6. At the end, rate your confidence: HIGH / MEDIUM / LOW

Answer:"""


TABLE_EXTRACTION_PROMPT = """Analyze this image which contains a table. Extract the data in a structured format.

Return the data as a JSON object with this structure:
{{
    "table_type": "financial|general|comparison",
    "headers": ["column1", "column2"],
    "rows": [
        ["value1", "value2"],
    ],
    "summary": "Brief description of what the table shows"
}}"""


CHART_ANALYSIS_PROMPT = """Analyze this chart/graph image. Describe:
1. The type of chart (bar, line, pie, etc.)
2. What data it represents
3. Key trends or values visible
4. Any labels, legends, or axis information

Provide a detailed text description that captures all the information in the chart."""


DOCUMENT_SUMMARY_PROMPT = """Summarize the following document content in 3-5 sentences. Focus on the main topics, key findings, and important data points.

Content:
{content}

Summary:"""
