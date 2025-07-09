#!/usr/bin/env python3
"""
Database module for Strategic Analysis API
Handles storage and retrieval of analysis results
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

class AnalysisDatabase:
    """Database handler for analysis results using SQLite"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        # Always use SQLite to avoid MySQL dependency issues
        self.db_path = db_path or os.environ.get('DATABASE_URL', 'analysis_database.db')
        
        # Clean up database URL if it contains SQLite prefix
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path.replace('sqlite:///', '')
        elif self.db_path.startswith('mysql://') or self.db_path.startswith('postgresql://'):
            # Fallback to SQLite if other database URLs are provided but not available
            logger.warning("Non-SQLite database URL detected, falling back to SQLite")
            self.db_path = 'analysis_database.db'
        
        self.lock = threading.Lock()
        
        try:
            self._create_tables()
            logger.info(f"✅ Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _create_tables(self):
        """Create necessary database tables"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,  -- JSON array
                        files_processed INTEGER DEFAULT 0,
                        business_strategy_dominant TEXT,
                        business_strategy_secondary TEXT,
                        business_strategy_reasoning TEXT,
                        risk_strategy_dominant TEXT,
                        risk_strategy_secondary TEXT,
                        risk_strategy_reasoning TEXT,
                        analysis_type TEXT DEFAULT 'unknown',
                        confidence_level TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'completed',
                        raw_response TEXT,  -- Full JSON response
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_company_number ON analyses(company_number)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_date ON analyses(analysis_date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_company_name ON analyses(company_name)')
                
                conn.commit()
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self._get_connection() as conn:
                conn.execute('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """
        Store analysis result in database
        
        Args:
            analysis_data: Analysis result dictionary
            
        Returns:
            int: ID of stored record
        """
        with self.lock:
            try:
                # Extract data from analysis_data
                company_number = analysis_data.get('company_number', '')
                company_name = analysis_data.get('company_name', '')
                analysis_date = analysis_data.get('analysis_date', datetime.now().isoformat())
                years_analyzed = json.dumps(analysis_data.get('years_analyzed', []))
                files_processed = analysis_data.get('files_processed', 0)
                
                # Extract business strategy
                business_strategy = analysis_data.get('business_strategy', {})
                business_strategy_dominant = business_strategy.get('dominant', '')
                business_strategy_secondary = business_strategy.get('secondary', '')
                business_strategy_reasoning = business_strategy.get('reasoning', '')
                
                # Extract risk strategy
                risk_strategy = analysis_data.get('risk_strategy', {})
                risk_strategy_dominant = risk_strategy.get('dominant', '')
                risk_strategy_secondary = risk_strategy.get('secondary', '')
                risk_strategy_reasoning = risk_strategy.get('reasoning', '')
                
                analysis_type = analysis_data.get('analysis_type', 'unknown')
                confidence_level = analysis_data.get('confidence_level', 'medium')
                status = analysis_data.get('status', 'completed')
                raw_response = json.dumps(analysis_data)
                
                with self._get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO analyses (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy_dominant, business_strategy_secondary,
                            business_strategy_reasoning, risk_strategy_dominant, risk_strategy_secondary,
                            risk_strategy_reasoning, analysis_type, confidence_level, status, raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_number, company_name, analysis_date, years_analyzed,
                        files_processed, business_strategy_dominant, business_strategy_secondary,
                        business_strategy_reasoning, risk_strategy_dominant, risk_strategy_secondary,
                        risk_strategy_reasoning, analysis_type, confidence_level, status, raw_response
                    ))
                    
                    record_id = cursor.lastrowid
                    conn.commit()
                    
                    logger.info(f"Analysis stored successfully with ID: {record_id}")
                    return record_id
                    
            except Exception as e:
                logger.error(f"Error storing analysis result: {e}")
                raise
    
    def get_analysis_by_company(self, company_number: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a company
        
        Args:
            company_number: Company number
            
        Returns:
            List of analysis records
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute('''
                    SELECT * FROM analyses 
                    WHERE company_number = ? 
                    ORDER BY analysis_date DESC
                ''', (company_number,)).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting analyses for company {company_number}: {e}")
            return []
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent analyses
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent analysis records
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute('''
                    SELECT * FROM analyses 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return []
    
    def search_companies(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search companies by name or number
        
        Args:
            search_term: Search term
            
        Returns:
            List of matching companies
        """
        try:
            search_pattern = f"%{search_term}%"
            
            with self._get_connection() as conn:
                rows = conn.execute('''
                    SELECT DISTINCT company_number, company_name, 
                           COUNT(*) as analysis_count,
                           MAX(analysis_date) as latest_analysis
                    FROM analyses 
                    WHERE company_name LIKE ? OR company_number LIKE ?
                    GROUP BY company_number, company_name
                    ORDER BY latest_analysis DESC
                ''', (search_pattern, search_pattern)).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
    
    def delete_analysis_by_id(self, analysis_id: int, company_number: str) -> bool:
        """
        Delete a specific analysis
        
        Args:
            analysis_id: Analysis ID
            company_number: Company number for safety check
            
        Returns:
            bool: True if deleted successfully
        """
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute('''
                        DELETE FROM analyses 
                        WHERE id = ? AND company_number = ?
                    ''', (analysis_id, company_number))
                    
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"Deleted analysis {analysis_id} for company {company_number}")
                        return True
                    else:
                        logger.warning(f"No analysis found with ID {analysis_id} for company {company_number}")
                        return False
                        
            except Exception as e:
                logger.error(f"Error deleting analysis {analysis_id}: {e}")
                return False
    
    def cleanup_invalid_analyses(self, company_number: str) -> int:
        """
        Clean up invalid analyses for a company
        CONSERVATIVE: Only removes analyses with multiple specific issues
        
        Args:
            company_number: Company number
            
        Returns:
            int: Number of analyses deleted
        """
        with self.lock:
            try:
                # Get all analyses for the company
                analyses = self.get_analysis_by_company(company_number)
                
                ids_to_delete = []
                
                for analysis in analyses:
                    issues = []
                    
                    # Check for specific issues
                    company_name = analysis.get('company_name', '')
                    if 'HSBC' in company_name and company_number == '02613335':
                        issues.append("wrong_company")
                    
                    business_reasoning = analysis.get('business_strategy_reasoning', '')
                    if business_reasoning == 'The company demonstrates strong growth-oriented strategies with focus on market expansion and innovation.':
                        issues.append("generic_business_reasoning")
                    
                    risk_reasoning = analysis.get('risk_strategy_reasoning', '')
                    if risk_reasoning == 'Conservative risk management approach with emphasis on regulatory compliance and stable operations.':
                        issues.append("generic_risk_reasoning")
                    
                    # Only delete if multiple issues (VERY CONSERVATIVE)
                    if len(issues) >= 2:
                        ids_to_delete.append(analysis['id'])
                
                # Delete identified analyses
                deleted_count = 0
                if ids_to_delete:
                    with self._get_connection() as conn:
                        for analysis_id in ids_to_delete:
                            cursor = conn.execute('''
                                DELETE FROM analyses 
                                WHERE id = ? AND company_number = ?
                            ''', (analysis_id, company_number))
                            
                            if cursor.rowcount > 0:
                                deleted_count += 1
                        
                        conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} invalid analyses for company {company_number}")
                return deleted_count
                
            except Exception as e:
                logger.error(f"Error cleaning up analyses for company {company_number}: {e}")
                return 0
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with self._get_connection() as conn:
                # Total analyses
                total_count = conn.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
                
                # Unique companies
                company_count = conn.execute('SELECT COUNT(DISTINCT company_number) FROM analyses').fetchone()[0]
                
                # Recent activity (last 30 days)
                recent_count = conn.execute('''
                    SELECT COUNT(*) FROM analyses 
                    WHERE created_at >= datetime('now', '-30 days')
                ''').fetchone()[0]
                
                # Analysis types
                type_stats = conn.execute('''
                    SELECT analysis_type, COUNT(*) 
                    FROM analyses 
                    GROUP BY analysis_type
                ''').fetchall()
                
                return {
                    'total_analyses': total_count,
                    'unique_companies': company_count,
                    'recent_analyses_30_days': recent_count,
                    'analysis_types': dict(type_stats),
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the database
    print("Testing AnalysisDatabase...")
    
    db = AnalysisDatabase('test_analysis.db')
    
    # Test connection
    if db.test_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
    
    # Test storing a sample analysis
    sample_analysis = {
        'company_number': '12345678',
        'company_name': 'Test Company Ltd',
        'analysis_date': datetime.now().isoformat(),
        'years_analyzed': [2023, 2024],
        'files_processed': 2,
        'business_strategy': {
            'dominant': 'Growth',
            'secondary': 'Innovation',
            'reasoning': 'Test reasoning'
        },
        'risk_strategy': {
            'dominant': 'Conservative',
            'secondary': 'Balanced',
            'reasoning': 'Test risk reasoning'
        },
        'analysis_type': 'test',
        'status': 'completed'
    }
    
    try:
        record_id = db.store_analysis_result(sample_analysis)
        print(f"✅ Sample analysis stored with ID: {record_id}")
        
        # Test retrieval
        results = db.get_analysis_by_company('12345678')
        print(f"✅ Retrieved {len(results)} analyses for test company")
        
        # Test statistics
        stats = db.get_analysis_statistics()
        print(f"✅ Database statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    print("Database test completed.")