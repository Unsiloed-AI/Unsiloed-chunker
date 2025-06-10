import json
from typing import List, Dict, Any, Union, Optional
import logging
import re
from Unsiloed.utils.openai import get_openai_client

logger = logging.getLogger(__name__)

class QueryType:
    """Enum-like class for query types"""
    SIMPLE = "simple"           # Direct retrieval
    MULTI_HOP = "multi_hop"     # Requires multiple retrievals
    NEGATION = "negation"       # Contains negation ("not", "doesn't", etc.)
    COMPARISON = "comparison"   # Compares multiple items/facts
    TEMPORAL = "temporal"       # Time-based reasoning
    UNKNOWN = "unknown"         # Could not classify

def identify_query_type(query: str) -> str:
    """Identify query type based on patterns and keywords.
    
    Args:
        query: The query string
        
    Returns:
        Query type from QueryType class
    """
    query = query.lower()
    
    # Check for negation patterns
    negation_patterns = [
        r'\bnot\b', r'\bno\b', r"n't\b", r'\bwithout\b', 
        r'\bexcept\b', r'\bexcluding\b', r'\bomit\b', r'\bomitting\b'
    ]
    for pattern in negation_patterns:
        if re.search(pattern, query):
            return QueryType.NEGATION
    
    # Check for multi-hop patterns
    multi_hop_patterns = [
        r'\band\b.*\bhow\b', r'\bthen\b', r'\bafter\b', r'\bbefore\b',
        r'\brelationship\b', r'\brelated\b', r'\bconnection\b', 
        r'\bimpact\b', r'\binfluence\b', r'\bcause\b.*\beffect\b',
        r'\bsteps\b', r'\bprocess\b', r'\bsequence\b'
    ]
    for pattern in multi_hop_patterns:
        if re.search(pattern, query):
            return QueryType.MULTI_HOP
    
    # Check for comparison patterns
    comparison_patterns = [
        r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b',
        r'\bsimilarity\b', r'\bbetter\b', r'\bworse\b', r'\badvantage\b',
        r'\bdisadvantage\b', r'\bpros\b.*\bcons\b'
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, query):
            return QueryType.COMPARISON
    
    # Check for temporal patterns
    temporal_patterns = [
        r'\bwhen\b', r'\bdate\b', r'\btime\b', r'\byear\b', 
        r'\bduring\b', r'\bperiod\b', r'\bera\b', r'\bcentury\b',
        r'\bdecade\b', r'\bhistory\b', r'\bevolution\b'
    ]
    for pattern in temporal_patterns:
        if re.search(pattern, query):
            return QueryType.TEMPORAL
    
    # Default to simple if no complex patterns detected
    return QueryType.SIMPLE

def analyze_query(query: str) -> Dict[str, Any]:
    """Analyze the query to determine its type and characteristics.
    
    Args:
        query: The user query string
        
    Returns:
        Dictionary with query analysis
    """
    try:
        # First use pattern matching for common query types
        query_type = identify_query_type(query)
        
        client = get_openai_client()
        if not client:
            logger.error("Failed to initialize OpenAI client")
            raise ValueError("OpenAI client initialization failed")
            
        # Use OpenAI to analyze the query in more detail
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert at analyzing search queries. Your task is to determine:
                    1. The type of query (simple, multi_hop, negation, comparison, temporal, or unknown)
                    2. Key entities mentioned in the query
                    3. If it's a multi-hop query, identify the logical steps required
                    4. If it's a negation query, identify what is being negated
                    
                    I believe this query is of type: {query_type}
                    
                    Return your analysis as a JSON object with the following structure:
                    {{
                        "type": "query_type_here",
                        "entities": ["entity1", "entity2"],
                        "steps": ["step1", "step2"],  // for multi-hop queries
                        "negated_entity": "entity"    // for negation queries
                    }}
                    """
                },
                {
                    "role": "user",
                    "content": f"Analyze this query and provide results as JSON: {query}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the response with error handling
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Add the original query to the result
            result["original_query"] = query
            
            # Default to our pattern-based type if API didn't return a valid type
            if "type" not in result or not result["type"] or result["type"] == "unknown":
                result["type"] = query_type
                
            logger.info(f"Analyzed query as {result.get('type', query_type)}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from OpenAI: {str(e)}")
            # Fallback to basic result with our pattern-based type
            return {
                "type": query_type,
                "original_query": query,
                "entities": [],
                "error": f"JSON parse error: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        # Return a basic analysis as fallback using pattern matching
        return {
            "type": identify_query_type(query),
            "original_query": query,
            "entities": [],
            "error": str(e)
        }

def decompose_query(query: str) -> List[Dict[str, Any]]:
    """Decompose a complex query into simpler sub-queries.
    
    Args:
        query: The complex query to decompose
        
    Returns:
        List of sub-query dictionaries
    """
    try:
        # First analyze the query
        analysis = analyze_query(query)
        
        # If it's a simple query, no need to decompose
        if analysis.get("type") == QueryType.SIMPLE:
            return [{
                "query": query,
                "type": QueryType.SIMPLE,
                "step": 1,
                "total_steps": 1
            }]
            
        # For complex queries, use LLM to decompose
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert at breaking down complex search queries into simpler steps.
                    For each step, provide:
                    1. The sub-query text that should be searched
                    2. How this step relates to the overall query
                    3. What to do with the information once retrieved
                    
                    This query is of type: {analysis.get("type", "unknown")}
                    
                    Break multi-hop queries into a sequence of simpler queries where each builds on the previous.
                    For negation queries, create steps to find positive examples, then filter out negated items.
                    
                    Return your analysis as a JSON object with a 'steps' array like this:
                    {{
                        "steps": [
                            {{
                                "query": "sub-query text",
                                "purpose": "what this step accomplishes",
                                "use": "how to use the retrieved information"
                            }},
                            ...
                        ]
                    }}
                    """
                },
                {
                    "role": "user",
                    "content": f"Break down this query into search steps and provide results as JSON: {query}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the response with error handling
        try:
            result = json.loads(response.choices[0].message.content)
            steps = result.get("steps", [])
            
            # Ensure each step has required fields
            for i, step in enumerate(steps):
                step["step"] = i + 1
                step["total_steps"] = len(steps)
                if "query" not in step:
                    step["query"] = step.get("sub_query", f"Step {i+1} for: {query}")
                    
            logger.info(f"Decomposed query into {len(steps)} steps")
            return steps
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from OpenAI: {str(e)}")
            # Return a single step as fallback
            return [{
                "query": query,
                "type": analysis.get("type", QueryType.UNKNOWN),
                "step": 1,
                "total_steps": 1,
                "error": f"JSON parse error: {str(e)}"
            }]
        
    except Exception as e:
        logger.error(f"Error decomposing query: {str(e)}")
        # Return the original query as fallback
        return [{
            "query": query,
            "type": QueryType.UNKNOWN,
            "step": 1,
            "total_steps": 1,
            "error": str(e)
        }]

def synthesize_results(sub_results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
    """Synthesize results from multiple sub-queries into a coherent answer.
    
    Args:
        sub_results: List of results from sub-queries
        original_query: The original complex query
        
    Returns:
        Synthesized result dictionary
    """
    try:
        client = get_openai_client()
        
        # Prepare the results for the LLM
        results_text = ""
        for i, result in enumerate(sub_results):
            results_text += f"\nStep {i+1} Results:\n"
            results_text += f"Query: {result.get('query', 'Unknown')}\n"
            
            # Include retrieved chunks
            chunks = result.get("chunks", [])
            for j, chunk in enumerate(chunks[:3]):  # Limit to top 3 chunks per step
                results_text += f"- Chunk {j+1}: {chunk.get('text', '')[:200]}...\n"
        
        # Use LLM to synthesize the results
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at synthesizing information from multiple sources.
                    Your task is to create a coherent answer to the original query based on the results from 
                    multiple search steps. Only use the information provided in the results.
                    If the information is insufficient, state what's missing.
                    """
                },
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\n\nSearch results: {results_text}\n\nPlease synthesize a complete answer."
                }
            ],
            temperature=0.3
        )
        
        synthesis = response.choices[0].message.content
        
        return {
            "original_query": original_query,
            "synthesized_answer": synthesis,
            "sub_results": sub_results
        }
        
    except Exception as e:
        logger.error(f"Error synthesizing results: {str(e)}")
        # Return a basic synthesis as fallback
        return {
            "original_query": original_query,
            "synthesized_answer": "Could not synthesize results due to an error.",
            "sub_results": sub_results,
            "error": str(e)
        } 