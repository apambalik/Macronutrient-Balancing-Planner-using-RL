"""
Feedback processing system for user feedback on meal plans.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field


@dataclass
class UserFeedback:
    """Represents user feedback on a meal plan."""
    user_id: str
    meal_plan_id: str
    timestamp: datetime
    satisfaction_score: float  # 0-1 scale
    nutrition_score: float    # 0-1 scale
    variety_score: float      # 0-1 scale
    overall_rating: float     # 1-5 scale
    comments: str = ""
    specific_feedback: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary."""
        return {
            'user_id': self.user_id,
            'meal_plan_id': self.meal_plan_id,
            'timestamp': self.timestamp.isoformat(),
            'satisfaction_score': self.satisfaction_score,
            'nutrition_score': self.nutrition_score,
            'variety_score': self.variety_score,
            'overall_rating': self.overall_rating,
            'comments': self.comments,
            'specific_feedback': self.specific_feedback
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Create feedback from dictionary."""
        return cls(
            user_id=data['user_id'],
            meal_plan_id=data['meal_plan_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            satisfaction_score=data['satisfaction_score'],
            nutrition_score=data['nutrition_score'],
            variety_score=data['variety_score'],
            overall_rating=data['overall_rating'],
            comments=data.get('comments', ''),
            specific_feedback=data.get('specific_feedback', {})
        )


class FeedbackProcessor:
    """
    Processes user feedback and updates the RL agent accordingly.
    """
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = feedback_dir
        self.logger = logging.getLogger(__name__)
        
        # Create feedback directory
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Load existing feedback
        self.feedback_history: Dict[str, List[UserFeedback]] = {}
        self._load_feedback_history()
    
    def _load_feedback_history(self):
        """Load feedback history from files."""
        if os.path.exists(os.path.join(self.feedback_dir, "feedback_history.json")):
            try:
                with open(os.path.join(self.feedback_dir, "feedback_history.json"), 'r') as f:
                    data = json.load(f)
                
                for user_id, feedback_list in data.items():
                    self.feedback_history[user_id] = [
                        UserFeedback.from_dict(feedback_data) 
                        for feedback_data in feedback_list
                    ]
                
                self.logger.info(f"Loaded feedback history for {len(self.feedback_history)} users")
            except Exception as e:
                self.logger.error(f"Error loading feedback history: {e}")
    
    def _save_feedback_history(self):
        """Save feedback history to files."""
        data = {}
        for user_id, feedback_list in self.feedback_history.items():
            data[user_id] = [feedback.to_dict() for feedback in feedback_list]
        
        with open(os.path.join(self.feedback_dir, "feedback_history.json"), 'w') as f:
            json.dump(data, f, indent=2)
    
    def process_feedback(self, user_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback and return processed feedback for agent update.
        
        Args:
            user_id: User identifier
            feedback_data: Raw feedback data
            
        Returns:
            Processed feedback for agent update
        """
        self.logger.info(f"Processing feedback for user: {user_id}")
        
        # Create feedback object
        feedback = UserFeedback(
            user_id=user_id,
            meal_plan_id=feedback_data.get('meal_plan_id', 'unknown'),
            timestamp=datetime.now(),
            satisfaction_score=feedback_data.get('satisfaction_score', 0.5),
            nutrition_score=feedback_data.get('nutrition_score', 0.5),
            variety_score=feedback_data.get('variety_score', 0.5),
            overall_rating=feedback_data.get('overall_rating', 3.0),
            comments=feedback_data.get('comments', ''),
            specific_feedback=feedback_data.get('specific_feedback', {})
        )
        
        # Store feedback
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        self.feedback_history[user_id].append(feedback)
        
        # Save to file
        self._save_feedback_history()
        
        # Process feedback for agent update
        processed_feedback = self._process_for_agent(feedback)
        
        self.logger.info(f"Feedback processed successfully for user: {user_id}")
        return processed_feedback
    
    def _process_for_agent(self, feedback: UserFeedback) -> Dict[str, Any]:
        """
        Process feedback for agent update.
        
        Args:
            feedback: User feedback object
            
        Returns:
            Processed feedback for agent update
        """
        # Calculate feedback scores
        satisfaction_score = feedback.satisfaction_score
        nutrition_score = feedback.nutrition_score
        variety_score = feedback.variety_score
        overall_score = feedback.overall_rating / 5.0  # Normalize to 0-1
        
        # Analyze specific feedback
        specific_insights = self._analyze_specific_feedback(feedback.specific_feedback)
        
        # Calculate weighted feedback score
        weighted_score = (
            0.3 * satisfaction_score +
            0.4 * nutrition_score +
            0.2 * variety_score +
            0.1 * overall_score
        )
        
        # Determine feedback type
        if weighted_score >= 0.8:
            feedback_type = "positive"
        elif weighted_score >= 0.6:
            feedback_type = "neutral"
        else:
            feedback_type = "negative"
        
        return {
            'user_id': feedback.user_id,
            'feedback_type': feedback_type,
            'weighted_score': weighted_score,
            'satisfaction_score': satisfaction_score,
            'nutrition_score': nutrition_score,
            'variety_score': variety_score,
            'overall_score': overall_score,
            'specific_insights': specific_insights,
            'timestamp': feedback.timestamp,
            'comments': feedback.comments
        }
    
    def _analyze_specific_feedback(self, specific_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze specific feedback for insights.
        
        Args:
            specific_feedback: Specific feedback data
            
        Returns:
            Analyzed insights
        """
        insights = {
            'liked_foods': [],
            'disliked_foods': [],
            'nutritional_issues': [],
            'variety_issues': [],
            'preference_updates': {}
        }
        
        # Analyze liked/disliked foods
        if 'liked_foods' in specific_feedback:
            insights['liked_foods'] = specific_feedback['liked_foods']
        
        if 'disliked_foods' in specific_feedback:
            insights['disliked_foods'] = specific_feedback['disliked_foods']
        
        # Analyze nutritional issues
        if 'nutritional_issues' in specific_feedback:
            insights['nutritional_issues'] = specific_feedback['nutritional_issues']
        
        # Analyze variety issues
        if 'variety_issues' in specific_feedback:
            insights['variety_issues'] = specific_feedback['variety_issues']
        
        # Analyze preference updates
        if 'preference_updates' in specific_feedback:
            insights['preference_updates'] = specific_feedback['preference_updates']
        
        return insights
    
    def get_user_feedback_history(self, user_id: str, days: int = 30) -> List[UserFeedback]:
        """
        Get feedback history for a user within specified days.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of feedback within the time window
        """
        if user_id not in self.feedback_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = [
            feedback for feedback in self.feedback_history[user_id]
            if feedback.timestamp >= cutoff_date
        ]
        
        return recent_feedback
    
    def get_feedback_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get feedback summary for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Feedback summary
        """
        recent_feedback = self.get_user_feedback_history(user_id, days)
        
        if not recent_feedback:
            return {
                'total_feedback': 0,
                'avg_satisfaction': 0.0,
                'avg_nutrition': 0.0,
                'avg_variety': 0.0,
                'avg_overall': 0.0,
                'feedback_trend': 'neutral'
            }
        
        # Calculate averages
        avg_satisfaction = sum(f.satisfaction_score for f in recent_feedback) / len(recent_feedback)
        avg_nutrition = sum(f.nutrition_score for f in recent_feedback) / len(recent_feedback)
        avg_variety = sum(f.variety_score for f in recent_feedback) / len(recent_feedback)
        avg_overall = sum(f.overall_rating for f in recent_feedback) / len(recent_feedback)
        
        # Determine trend
        if len(recent_feedback) >= 2:
            recent_avg = (avg_satisfaction + avg_nutrition + avg_variety) / 3
            older_feedback = recent_feedback[:-1]
            older_avg = sum((f.satisfaction_score + f.nutrition_score + f.variety_score) / 3 
                           for f in older_feedback) / len(older_feedback)
            
            if recent_avg > older_avg + 0.1:
                trend = 'improving'
            elif recent_avg < older_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_feedback': len(recent_feedback),
            'avg_satisfaction': avg_satisfaction,
            'avg_nutrition': avg_nutrition,
            'avg_variety': avg_variety,
            'avg_overall': avg_overall,
            'feedback_trend': trend
        }
    
    def get_global_feedback_insights(self) -> Dict[str, Any]:
        """
        Get global insights from all user feedback.
        
        Returns:
            Global feedback insights
        """
        all_feedback = []
        for feedback_list in self.feedback_history.values():
            all_feedback.extend(feedback_list)
        
        if not all_feedback:
            return {
                'total_users': 0,
                'total_feedback': 0,
                'avg_satisfaction': 0.0,
                'avg_nutrition': 0.0,
                'avg_variety': 0.0,
                'common_issues': [],
                'popular_foods': []
            }
        
        # Calculate global averages
        avg_satisfaction = sum(f.satisfaction_score for f in all_feedback) / len(all_feedback)
        avg_nutrition = sum(f.nutrition_score for f in all_feedback) / len(all_feedback)
        avg_variety = sum(f.variety_score for f in all_feedback) / len(all_feedback)
        
        # Analyze common issues
        common_issues = self._analyze_common_issues(all_feedback)
        
        # Analyze popular foods
        popular_foods = self._analyze_popular_foods(all_feedback)
        
        return {
            'total_users': len(self.feedback_history),
            'total_feedback': len(all_feedback),
            'avg_satisfaction': avg_satisfaction,
            'avg_nutrition': avg_nutrition,
            'avg_variety': avg_variety,
            'common_issues': common_issues,
            'popular_foods': popular_foods
        }
    
    def _analyze_common_issues(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Analyze common issues from feedback."""
        issues = []
        
        # Check for low satisfaction
        low_satisfaction_count = sum(1 for f in feedback_list if f.satisfaction_score < 0.5)
        if low_satisfaction_count > len(feedback_list) * 0.2:
            issues.append("low_satisfaction")
        
        # Check for nutrition issues
        low_nutrition_count = sum(1 for f in feedback_list if f.nutrition_score < 0.6)
        if low_nutrition_count > len(feedback_list) * 0.15:
            issues.append("nutrition_imbalance")
        
        # Check for variety issues
        low_variety_count = sum(1 for f in feedback_list if f.variety_score < 0.5)
        if low_variety_count > len(feedback_list) * 0.2:
            issues.append("lack_of_variety")
        
        return issues
    
    def _analyze_popular_foods(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Analyze popular foods from feedback."""
        # This would typically analyze specific feedback for food preferences
        # For now, return empty list
        return [] 