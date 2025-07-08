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

logger = logging.getLogger(__name__)

# DEBUG: Log environment variables immediately when module loads
logger.info("=== DATABASE MODULE LOADING DEBUG ===")
logger.info(f"DB_HOST from env: {os.getenv('DB_HOST')}")
logger.info(f"DB_NAME from env: {os.getenv('DB_NAME')}")
logger.info(f"DB_USER from env: {os.getenv('DB_USER')}")
logger.info(f"DB_PASSWORD set: {'YES' if os.getenv('DB_PASSWORD') else 'NO'}")
logger.info(f"DB_PORT from env: {os.getenv('DB_PORT', 3306)}")
logger.info("=== END MODULE LOADING DEBUG ===")

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

class AnalysisDatabase:
    """Handles database operations for analysis results"""
    
    def __init__(self):
        """Initialize database connection"""
        self.config = DB_CONFIG
        
        # Additional debug for the final config
        logger.info("=== DATABASE CONFIG DEBUG ===")
        logger.info(f"Final config host: {self.config['host']}")
        logger.info(f"Final config user: {self.config['user']}")
        logger.info(f"Final config database: {self.config['database']}")
        logger.info("=== END CONFIG DEBUG ===")
        
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
    
    def delete_analysis_by_id(self, analysis_id: int, company_number: str = None) -> bool:
        """
        Delete a specific analysis by ID
        
        Args:
            analysis_id: ID of the analysis to delete
            company_number: Optional company number for additional validation
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Optional: Verify the analysis belongs to the company
                if company_number:
                    verify_query = "SELECT id FROM analysis_results WHERE id = %s AND company_number = %s"
                    cursor.execute(verify_query, (analysis_id, company_number))
                    if not cursor.fetchone():
                        logger.warning(f"Analysis {analysis_id} not found for company {company_number}")
                        return False
                
                # Delete from analysis_history table first (foreign key constraint)
                history_delete_query = "DELETE FROM analysis_history WHERE analysis_id = %s"
                cursor.execute(history_delete_query, (analysis_id,))
                history_rows = cursor.rowcount
                logger.info(f"Deleted {history_rows} history records for analysis {analysis_id}")
                
                # Delete from main analysis_results table
                main_delete_query = "DELETE FROM analysis_results WHERE id = %s"
                cursor.execute(main_delete_query, (analysis_id,))
                main_rows = cursor.rowcount
                
                if main_rows > 0:
                    logger.info(f"Successfully deleted analysis {analysis_id}")
                    return True
                else:
                    logger.warning(f"No analysis found with ID {analysis_id}")
                    return False
                    
        except Error as e:
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            return False
    
    def cleanup_invalid_analyses(self, company_number: str) -> int:
        """
        Remove all invalid analysis entries for a company
        
        Args:
            company_number: Company registration number
            
        Returns:
            int: Number of analyses deleted
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                # Get all analyses for the company
                query = "SELECT * FROM analysis_results WHERE company_number = %s"
                cursor.execute(query, (company_number,))
                analyses = cursor.fetchall()
                
                deleted_count = 0
                
                for analysis in analyses:
                    is_invalid = False
                    reasons = []
                    
                    # Parse raw_response if it's a string
                    raw_response = analysis.get('raw_response')
                    if isinstance(raw_response, str):
                        try:
                            raw_response = json.loads(raw_response)
                        except:
                            raw_response = {}
                    
                    # Check for wrong company name patterns
                    company_name = analysis.get('company_name', '')
                    if 'HSBC' in company_name and company_number == '02613335':
                        is_invalid = True
                        reasons.append("Wrong company name (HSBC)")
                    
                    # Check for generic reasoning
                    business_reasoning = analysis.get('business_strategy_reasoning', '')
                    if 'demonstrates strong growth-oriented strategies' in business_reasoning:
                        is_invalid = True
                        reasons.append("Generic business reasoning")
                    
                    risk_reasoning = analysis.get('risk_strategy_reasoning', '')
                    if 'Conservative risk management approach' in risk_reasoning:
                        is_invalid = True
                        reasons.append("Generic risk reasoning")
                    
                    # Check for incomplete raw_response
                    if not raw_response or not isinstance(raw_response, dict):
                        is_invalid = True
                        reasons.append("Missing or invalid raw_response")
                    elif not raw_response.get('business_strategy') or not raw_response.get('risk_strategy'):
                        is_invalid = True
                        reasons.append("Incomplete raw_response structure")
                    
                    # Check reasoning length
                    if len(business_reasoning) < 100:
                        is_invalid = True
                        reasons.append("Business reasoning too short")
                    
                    if len(risk_reasoning) < 100:
                        is_invalid = True
                        reasons.append("Risk reasoning too short")
                    
                    if is_invalid:
                        logger.info(f"Deleting invalid analysis ID {analysis['id']}: {reasons}")
                        if self.delete_analysis_by_id(analysis['id'], company_number):
                            deleted_count += 1
                
                logger.info(f"Cleanup completed: deleted {deleted_count} invalid analyses for company {company_number}")
                return deleted_count
                
        except Error as e:
            logger.error(f"Error during cleanup for company {company_number}: {e}")
            return 0
    
    def get_analysis_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor(dictionary=True)
                
                stats = {}
                
                # Total analyses
                cursor.execute("SELECT COUNT(*) as total FROM analysis_results")
                stats['total_analyses'] = cursor.fetchone()['total']
                
                # Completed analyses
                cursor.execute("SELECT COUNT(*) as completed FROM analysis_results WHERE status = 'completed'")
                stats['completed_analyses'] = cursor.fetchone()['completed']
                
                # Unique companies
                cursor.execute("SELECT COUNT(DISTINCT company_number) as companies FROM analysis_results")
                stats['unique_companies'] = cursor.fetchone()['companies']
                
                # Recent analyses (last 7 days)
                cursor.execute("""
                    SELECT COUNT(*) as recent 
                    FROM analysis_results 
                    WHERE analysis_date >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)
                stats['recent_analyses'] = cursor.fetchone()['recent']
                
                logger.info(f"Database statistics: {stats}")
                return stats
                
        except Error as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}


# Test the database connection
if __name__ == "__main__":
    db = AnalysisDatabase()
    success = db.test_connection()
    if success:
        print("✅ Database connection successful!")
        
        # Show statistics
        stats = db.get_analysis_statistics()
        if stats:
            print(f"📊 Database Statistics:")
            print(f"   Total analyses: {stats.get('total_analyses', 0)}")
            print(f"   Completed: {stats.get('completed_analyses', 0)}")
            print(f"   Unique companies: {stats.get('unique_companies', 0)}")
            print(f"   Recent (7 days): {stats.get('recent_analyses', 0)}")
    else:
        print("❌ Database connection failed!")
        print("Check your environment variables:")
        print(f"DB_HOST: {os.getenv('DB_HOST')}")
        print(f"DB_NAME: {os.getenv('DB_NAME')}")
        print(f"DB_USER: {os.getenv('DB_USER')}")
        print(f"DB_PASSWORD: {'*' * len(os.getenv('DB_PASSWORD', '')) if os.getenv('DB_PASSWORD') else 'Not set'}")