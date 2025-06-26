import json
from datetime import datetime
from pgvector_db_setup import get_db_connection
from sqlalchemy import text
import streamlit as st
import os

def send_to_lambda(index, document):
    """
    Store UBI (User Behavior Insights) data in pgvector instead of Lambda
    This maintains the same interface but stores locally
    """
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            if index == "ubi_queries":
                # Store query data
                conn.execute(
                    text("""
                        INSERT INTO ubi_queries (
                            client_id, query_id, application, 
                            query_response_hit_ids, timestamp, 
                            user_query, query
                        ) VALUES (
                            :client_id, :query_id, :application,
                            :hit_ids, :timestamp,
                            :user_query, :query
                        )
                        ON CONFLICT (query_id) DO UPDATE SET
                            query_response_hit_ids = EXCLUDED.query_response_hit_ids,
                            timestamp = EXCLUDED.timestamp
                    """),
                    {
                        "client_id": document.get("client_id"),
                        "query_id": document.get("query_id"),
                        "application": document.get("application"),
                        "hit_ids": document.get("query_response_hit_ids", []),
                        "timestamp": document.get("timestamp"),
                        "user_query": json.dumps(document.get("user_query", {})),
                        "query": document.get("query")
                    }
                )
            
            elif index == "ubi_events":
                # Store event data
                conn.execute(
                    text("""
                        INSERT INTO ubi_events (
                            action_name, item_id, query_id,
                            session_id, timestamp, message_type,
                            message
                        ) VALUES (
                            :action_name, :item_id, :query_id,
                            :session_id, :timestamp, :message_type,
                            :message
                        )
                    """),
                    {
                        "action_name": document.get("action_name"),
                        "item_id": document.get("item_id"),
                        "query_id": document.get("query_id"),
                        "session_id": document.get("session_id"),
                        "timestamp": document.get("timestamp"),
                        "message_type": document.get("message_type"),
                        "message": document.get("message")
                    }
                )
            
            elif index == "otel-v1-apm-span-default":
                # Store OpenTelemetry traces (optional)
                # You can create a separate table for this if needed
                pass
            
            conn.commit()
        
        return 202  # Success status code
        
    except Exception as e:
        print(f"Error storing UBI data: {e}")
        return 500  # Error status code

def get_user_behavior_insights(session_id=None, time_range=None):
    """
    Retrieve user behavior insights from pgvector
    """
    engine = get_db_connection()
    
    with engine.connect() as conn:
        # Build query based on filters
        query = """
            SELECT q.*, 
                   COUNT(DISTINCT e.action_name) as event_count,
                   array_agg(DISTINCT e.action_name) as actions
            FROM ubi_queries q
            LEFT JOIN ubi_events e ON q.query_id = e.query_id
            WHERE 1=1
        """
        
        params = {}
        
        if session_id:
            query += " AND q.client_id = :session_id"
            params["session_id"] = session_id
        
        if time_range:
            query += " AND q.timestamp >= :start_time AND q.timestamp <= :end_time"
            params["start_time"] = time_range[0]
            params["end_time"] = time_range[1]
        
        query += " GROUP BY q.id ORDER BY q.timestamp DESC"
        
        result = conn.execute(text(query), params)
        
        insights = []
        for row in result:
            insights.append({
                "query_id": row.query_id,
                "query": row.query,
                "timestamp": row.timestamp,
                "hit_count": len(row.query_response_hit_ids) if row.query_response_hit_ids else 0,
                "event_count": row.event_count,
                "actions": row.actions
            })
        
        return insights

def get_popular_queries(limit=10):
    """
    Get most popular queries
    """
    engine = get_db_connection()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT query, COUNT(*) as count
                FROM ubi_queries
                WHERE query IS NOT NULL
                GROUP BY query
                ORDER BY count DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        
        return [{"query": row.query, "count": row.count} for row in result]

def get_click_through_rate(time_range=None):
    """
    Calculate click-through rate for queries
    """
    engine = get_db_connection()
    
    with engine.connect() as conn:
        query = """
            SELECT 
                COUNT(DISTINCT q.query_id) as total_queries,
                COUNT(DISTINCT e.query_id) as queries_with_clicks,
                COUNT(DISTINCT e.id) as total_clicks
            FROM ubi_queries q
            LEFT JOIN ubi_events e ON q.query_id = e.query_id 
                AND e.action_name = 'expander_open'
        """
        
        params = {}
        
        if time_range:
            query += " WHERE q.timestamp >= :start_time AND q.timestamp <= :end_time"
            params["start_time"] = time_range[0]
            params["end_time"] = time_range[1]
        
        result = conn.execute(text(query), params).fetchone()
        
        if result and result.total_queries > 0:
            ctr = result.queries_with_clicks / result.total_queries
            return {
                "click_through_rate": ctr,
                "total_queries": result.total_queries,
                "queries_with_clicks": result.queries_with_clicks,
                "total_clicks": result.total_clicks
            }
        
        return {
            "click_through_rate": 0,
            "total_queries": 0,
            "queries_with_clicks": 0,
            "total_clicks": 0
        }
