#!/usr/bin/env python3
"""
Database integration for storing and retrieving analysis results
Add this as database.py in your project root
"""

import mysql.connector
from mysql.connector import Error
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os
from contextlib import contextmanager

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'use_unicode': True,
    'autocommit': True
}

logger = logging.getLogger(__name__)

class AnalysisDatabase:
    """Handles database operations for analysis results"""
    
    def __init__(self):
        """Initialize database connection"""
        self.config = DB_CONFIG
        logger.info("Database configuration initialized")
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            connection = mysql.connector.connect(**self.config)
            logger.debug("Database connection established")
            yield connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
                logger.debug("Database connection closed")
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("✅ Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def store_analysis_result(self, analysis_data: Dict) -> int:
        """
        Store analysis result in database
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            int: ID of the stored record
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Prepare the data
                insert_query = """
                INSERT INTO analysis_results (
                    company_number, company_name, years_analyzed, files_processed,
                    business_strategy_dominant, business_strategy_reasoning,
                    risk_strategy_dominant, risk_strategy_reasoning,
                    status, raw_response
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Handle JSON serialization for older MySQL versions
                years_json = json.dumps(analysis_data.get('years_analyzed', []))
                raw_json = json.dumps(analysis_data)
                
                values = (
                    analysis_data.get('company_number'),
                    analysis_data.get('company_name'),
                    years_json,
                    analysis_data.get('files_processed', 0),
                    analysis_data.get('business_strategy', {}).get('dominant'),
                    analysis_data.get('business_strategy', {}).get('reasoning'),
                    analysis_data.get('risk_strategy', {}).get('dominant'),
                    analysis_data.get('risk_strategy', {}).get('reasoning'),
                    'completed' if analysis_data.get('success') else 'failed',
                    raw_json
                )
                
                cursor.execute(insert_query, values)
                record_id = cursor.lastrowid
                
                # Store in history table as well
                self._store_analysis_history(cursor, analysis_data, record_id)
                
                logger.info(f"Stored analysis result with ID: {record_id}")
                return record_id
                
        except Error as e:
            logger.error(f"Error storing analysis result: {e}")
            raise
    
    def _store_analysis_history(self, cursor, analysis_data: Dict, analysis_id: int):
        """Store analysis in history table"""
        try:
            history_query = """
            INSERT INTO analysis_history (
                company_number, analysis_id, years_analyzed,
                business_strategy, risk_strategy
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            years_json = json.dumps(analysis_data.get('years_analyzed', []))
            
            history_values = (
                analysis_data.get('company_number'),
                analysis_id,
                years_json,
                analysis_data.get('business_strategy', {}).get('dominant'),
                analysis_data.get('risk_strategy', {}).get('dominant')
            )
            
            cursor.execute(history_query, history_values)
            logger.debug("Analysis history stored successfully")
            
        except Error as e:
            logger.error(f"Error storing analysis history: {e}")
            # Don't raise - history is optional
    
    def get_analysis_by_company(self, company_number: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve analysis results for a company
        
        Args:
            company_number: Company registration number
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                query = """
                SELECT * FROM analysis_results 
                WHERE company_number = %s 
                ORDER BY analysis_date DESC 
                LIMIT %s
                """
                
                cursor.execute(query, (company_number, limit))
                results = cursor.fetchall()
                
                # Parse JSON fields
                for result in results:
                    if result['years_analyzed']:
                        try:
                            result['years_analyzed'] = json.loads(result['years_analyzed'])
                        except:
                            result['years_analyzed'] = []
                    if result['raw_response']:
                        try:
                            result['raw_response'] = json.loads(result['raw_response'])
                        except:
                            result['raw_response'] = {}
                
                logger.info(f"Retrieved {len(results)} analysis results for company {company_number}")
                return results
                
        except Error as e:
            logger.error(f"Error retrieving analysis results: {e}")
            return []
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict]:
        """Get most recent analyses across all companies"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                query = """
                SELECT company_number, company_name, analysis_date,
                       business_strategy_dominant, risk_strategy_dominant,
                       files_processed
                FROM analysis_results 
                WHERE status = 'completed'
                ORDER BY analysis_date DESC 
                LIMIT %s
                """
                
                cursor.execute(query, (limit,))
                results = cursor.fetchall()
                
                logger.info(f"Retrieved {len(results)} recent analyses")
                return results
                
        except Error as e:
            logger.error(f"Error retrieving recent analyses: {e}")
            return []
    
    def search_companies(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search companies by name or number
        
        Args:
            search_term: Company name or number to search for
            limit: Maximum results to return
            
        Returns:
            List of matching companies with their latest analysis
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                query = """
                SELECT DISTINCT company_number, company_name,
                       MAX(analysis_date) as latest_analysis,
                       COUNT(*) as analysis_count
                FROM analysis_results 
                WHERE company_name LIKE %s OR company_number LIKE %s
                GROUP BY company_number, company_name
                ORDER BY latest_analysis DESC
                LIMIT %s
                """
                
                search_pattern = f"%{search_term}%"
                cursor.execute(query, (search_pattern, search_pattern, limit))
                results = cursor.fetchall()
                
                logger.info(f"Found {len(results)} companies matching '{search_term}'")
                return results
                
        except Error as e:
            logger.error(f"Error searching companies: {e}")
            return []


# Test the database connection
if __name__ == "__main__":
    db = AnalysisDatabase()
    success = db.test_connection()
    if success:
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
        print("Check your environment variables:")
        print(f"DB_HOST: {os.getenv('DB_HOST')}")
        print(f"DB_NAME: {os.getenv('DB_NAME')}")
        print(f"DB_USER: {os.getenv('DB_USER')}")
        print(f"DB_PASSWORD: {'*' * len(os.getenv('DB_PASSWORD', '')) if os.getenv('DB_PASSWORD') else 'Not set'}")