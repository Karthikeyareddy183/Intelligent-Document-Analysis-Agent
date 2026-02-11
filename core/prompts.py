"""Prompt templates for LLM interactions."""

DOCUMENT_QA_PROMPT = """You are an expert document analyst with deep expertise in interpreting text, tables, charts, diagrams, and images from documents.

Answer the question based ONLY on the provided context chunks. Each chunk is labeled with its page number, content type, and image index (if applicable).

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context. Do not make up information.
2. Cite sources using [Page X] format for every fact you state.
3. If the question asks about a specific page, focus your answer on chunks from that page.
4. If the question asks about a diagram/image/figure, look for chunks with Type: image and reference the image description.
5. If data comes from a table, mention "Table on Page X" explicitly.
6. If you cannot answer from the context, say "Insufficient context to answer this question."
7. Be detailed and thorough — the user may be asking about a specific visual element.
8. At the end, rate your confidence: HIGH / MEDIUM / LOW

Answer:"""


IMAGE_DESCRIPTION_PROMPT = """Describe this image/diagram/figure from a document in detail. Include:
1. What type of visual this is (chart, diagram, flowchart, table, photograph, illustration, etc.)
2. All text labels, legends, axis labels, and annotations visible
3. The key information or data it conveys
4. Relationships between elements (if it's a flowchart or diagram)
5. Any numerical values, percentages, or trends visible

Provide a thorough text description that would allow someone who cannot see the image to fully understand its content. Be factual and precise."""


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
