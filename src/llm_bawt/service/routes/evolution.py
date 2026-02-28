"""Evolution monitoring and bot personality analytics routes.

This module provides deep analytics into bot performance, personality evolution,
and conversation quality - enabling data-driven bot improvement.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_service
from ..turn_logs import TurnLogStore

router = APIRouter(prefix="/v1/evolution", tags=["Evolution"])


# =============================================================================
# Schemas
# =============================================================================

class BotEvolutionMetrics(BaseModel):
    """Comprehensive evolution metrics for a bot."""
    
    bot_id: str
    period_hours: int
    
    # Volume metrics
    total_turns: int
    unique_users: int
    total_messages: int
    
    # Performance
    avg_latency_ms: float
    p95_latency_ms: float
    streaming_usage_rate: float
    
    # Engagement quality
    avg_turns_per_session: float
    user_return_rate: float  # Users with multiple sessions
    
    # Tool usage
    total_tool_calls: int
    tool_calls_per_turn: float
    top_tools: list[dict[str, Any]]
    
    # Content analysis
    avg_response_length: int
    question_asking_rate: float  # Bot asks follow-up questions
    
    # Quality indicators
    error_rate: float
    user_satisfaction_proxy: float  # Based on follow-up patterns
    
    # Personality markers (extracted from response analysis)
    personality_traits: dict[str, float] | None = None
    topic_affinity: dict[str, int] | None = None


class ConversationQualityScore(BaseModel):
    """Quality scoring for a specific conversation."""
    
    bot_id: str
    user_id: str
    session_id: str | None = None
    
    # Engagement metrics
    turn_count: int
    session_duration_minutes: float | None = None
    
    # Quality signals
    engagement_score: float  # 0-1 based on turn depth and variety
    coherence_score: float  # 0-1 based on context maintenance
    personality_consistency: float  # 0-1 based on style consistency
    
    # Improvement flags
    flags: list[str]  # e.g., ["abrupt_end", "repetitive", "unclear_response"]
    
    # Raw signals that fed into scores
    signals: dict[str, Any] | None = None


class PersonalityDriftReport(BaseModel):
    """Report on how a bot's personality has evolved over time."""
    
    bot_id: str
    baseline_period_days: int = 7
    comparison_period_days: int = 1
    
    drift_detected: bool
    drift_magnitude: float  # 0-1, higher = more drift
    confidence: float
    
    specific_changes: list[dict[str, Any]]  # What specifically changed
    
    # Temporal comparison
    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]
    
    # Recommendations
    recommended_adjustments: list[str]


class ImprovementSuggestion(BaseModel):
    """Actionable improvement suggestion."""
    
    id: str
    priority: str  # critical, high, medium, low
    category: str  # performance, engagement, personality, reliability
    
    bot_id: str
    user_id: str | None = None  # Null = applies to all users
    
    title: str
    description: str
    
    # Metrics that triggered this
    triggering_metrics: dict[str, Any] | None = None
    
    # Suggested actions
    suggested_prompt_adjustments: list[str] | None = None
    suggested_config_changes: dict[str, Any] | None = None
    
    created_at: str
    acknowledged: bool = False


class EvolutionDashboardSummary(BaseModel):
    """High-level summary for the evolution dashboard."""
    
    generated_at: str
    period_hours: int
    
    # System overview
    total_turns: int
    total_bots: int
    active_bot_user_pairs: int
    
    # Health indicators
    avg_system_latency: float
    error_rate: float
    
    # Top insights
    top_performing_bot: str | None = None
    bot_needing_attention: str | None = None
    
    # Suggestions summary
    critical_suggestions: int
    high_suggestions: int
    medium_suggestions: int
    
    recent_suggestions: list[ImprovementSuggestion]


# =============================================================================
# Analytics Engine
# =============================================================================

class EvolutionAnalytics:
    """Core analytics engine for bot evolution tracking."""
    
    def __init__(self, turn_store: TurnLogStore) -> None:
        self.store = turn_store
    
    def analyze_turns(
        self,
        bot_id: str | None = None,
        user_id: str | None = None,
        since_hours: int = 24
    ) -> dict[str, Any]:
        """Analyze turn logs for evolution metrics."""
        
        # Get raw turn data
        rows, total = self.store.list_turns(
            bot_id=bot_id,
            user_id=user_id,
            since_hours=since_hours,
            limit=1000,
            offset=0
        )
        
        if not rows:
            return {"empty": True, "total_turns": 0}
        
        # Group by bot-user pairs
        pairs: dict[tuple[str, str], list] = defaultdict(list)
        for row in rows:
            key = (row.bot_id or "unknown", row.user_id or "unknown")
            pairs[key].append(row)
        
        # Calculate metrics per pair
        pair_metrics = {}
        for (b_id, u_id), turns in pairs.items():
            metrics = self._analyze_conversation_pair(turns)
            pair_metrics[f"{b_id}:{u_id}"] = metrics
        
        return {
            "empty": False,
            "total_turns": len(rows),
            "unique_pairs": len(pairs),
            "pair_metrics": pair_metrics,
            "aggregated": self._aggregate_metrics(pair_metrics)
        }
    
    def _analyze_conversation_pair(self, turns: list) -> dict[str, Any]:
        """Analyze metrics for a single bot-user conversation pair."""
        
        # Basic counts
        total = len(turns)
        latencies = [t.latency_ms for t in turns if t.latency_ms]
        tool_counts = [len(json.loads(t.tool_calls_json or "[]")) for t in turns]
        
        # Response length estimation
        response_lengths = []
        for t in turns:
            if t.response_text:
                response_lengths.append(len(t.response_text))
        
        # Streaming usage
        streaming_count = sum(1 for t in turns if t.stream)
        
        # Error analysis (status other than 'ok')
        error_count = sum(1 for t in turns if t.status and t.status not in ("ok", "success"))
        
        # Engagement patterns (time gaps indicate session boundaries)
        timestamps = [t.created_at for t in turns if t.created_at]
        session_count = self._estimate_sessions(timestamps)
        
        return {
            "total_turns": total,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
            "streaming_rate": streaming_count / total if total else 0,
            "error_rate": error_count / total if total else 0,
            "avg_tool_calls": sum(tool_counts) / total if total else 0,
            "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "estimated_sessions": session_count,
            "avg_turns_per_session": total / session_count if session_count else 0,
        }
    
    def _estimate_sessions(self, timestamps: list[str]) -> int:
        """Estimate number of conversation sessions based on time gaps."""
        if not timestamps:
            return 0
        
        # Sort timestamps
        sorted_ts = sorted(timestamps)
        
        # Session boundary = 30 minute gap
        session_count = 1
        for i in range(1, len(sorted_ts)):
            try:
                t1 = datetime.fromisoformat(sorted_ts[i-1].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(sorted_ts[i].replace("Z", "+00:00"))
                gap = (t2 - t1).total_seconds() / 60  # minutes
                if gap > 30:
                    session_count += 1
            except Exception:
                pass
        
        return session_count
    
    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _aggregate_metrics(self, pair_metrics: dict[str, Any]) -> dict[str, Any]:
        """Aggregate metrics across all pairs."""
        
        all_latencies = []
        all_error_rates = []
        all_tool_rates = []
        
        for metrics in pair_metrics.values():
            all_latencies.append(metrics["avg_latency_ms"])
            all_error_rates.append(metrics["error_rate"])
            all_tool_rates.append(metrics["avg_tool_calls"])
        
        return {
            "system_avg_latency": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
            "system_error_rate": sum(all_error_rates) / len(all_error_rates) if all_error_rates else 0,
            "system_avg_tool_rate": sum(all_tool_rates) / len(all_tool_rates) if all_tool_rates else 0,
            "total_pairs_analyzed": len(pair_metrics)
        }
    
    def detect_personality_drift(
        self,
        bot_id: str,
        baseline_days: int = 7,
        current_days: int = 1
    ) -> dict[str, Any]:
        """Detect if bot personality has drifted from baseline."""
        
        # Get baseline metrics
        baseline = self.analyze_turns(
            bot_id=bot_id,
            since_hours=baseline_days * 24
        )
        
        # Get current metrics  
        current = self.analyze_turns(
            bot_id=bot_id,
            since_hours=current_days * 24
        )
        
        if baseline.get("empty") or current.get("empty"):
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        base_agg = baseline["aggregated"]
        curr_agg = current["aggregated"]
        
        changes = []
        drift_score = 0.0
        
        # Check response length change
        base_len = base_agg.get("avg_response_length", 0)
        curr_len = curr_agg.get("avg_response_length", 0)
        if base_len > 0 and abs(curr_len - base_len) / base_len > 0.3:
            changes.append({
                "type": "response_length",
                "direction": "increased" if curr_len > base_len else "decreased",
                "baseline": base_len,
                "current": curr_len,
                "change_pct": abs(curr_len - base_len) / base_len * 100
            })
            drift_score += 0.2
        
        # Check latency change (may indicate model/personality change)
        base_lat = base_agg.get("system_avg_latency", 0)
        curr_lat = curr_agg.get("system_avg_latency", 0)
        if base_lat > 0 and abs(curr_lat - base_lat) / base_lat > 0.5:
            changes.append({
                "type": "latency",
                "direction": "increased" if curr_lat > base_lat else "decreased",
                "baseline": base_lat,
                "current": curr_lat
            })
            drift_score += 0.1
        
        # Check tool usage change
        base_tools = base_agg.get("system_avg_tool_rate", 0)
        curr_tools = curr_agg.get("system_avg_tool_rate", 0)
        if base_tools > 0 and abs(curr_tools - base_tools) / base_tools > 0.5:
            changes.append({
                "type": "tool_usage",
                "direction": "increased" if curr_tools > base_tools else "decreased",
                "baseline": base_tools,
                "current": curr_tools
            })
            drift_score += 0.2
        
        return {
            "drift_detected": drift_score > 0.3,
            "drift_magnitude": min(drift_score, 1.0),
            "confidence": 0.7 if changes else 0.3,
            "changes": changes,
            "baseline_metrics": base_agg,
            "current_metrics": curr_agg,
            "recommended_adjustments": self._generate_drift_recommendations(changes)
        }
    
    def _generate_drift_recommendations(self, changes: list[dict]) -> list[str]:
        """Generate recommendations based on detected drift."""
        
        recs = []
        for change in changes:
            if change["type"] == "response_length":
                if change["direction"] == "increased":
                    recs.append("Consider adding length constraints to system prompt")
                else:
                    recs.append("Responses getting shorter - may indicate confidence issues")
            
            elif change["type"] == "tool_usage":
                if change["direction"] == "increased":
                    recs.append("Review tool necessity - may be over-tooling")
                else:
                    recs.append("Tool usage declining - may be losing capability utilization")
        
        return recs
    
    def generate_suggestions(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate improvement suggestions from analysis."""
        
        suggestions = []
        
        if analysis.get("empty"):
            return suggestions
        
        aggregated = analysis.get("aggregated", {})
        pair_metrics = analysis.get("pair_metrics", {})
        
        # System-level suggestions
        sys_latency = aggregated.get("system_avg_latency", 0)
        if sys_latency > 5000:
            suggestions.append({
                "id": f"perf-{int(time.time())}",
                "priority": "high",
                "category": "performance",
                "bot_id": "system",
                "title": "High System Latency",
                "description": f"Average latency of {sys_latency:.0f}ms exceeds 5s threshold. Consider model optimization or caching.",
                "triggering_metrics": {"avg_latency_ms": sys_latency}
            })
        
        # Per-pair suggestions
        for pair_key, metrics in pair_metrics.items():
            bot_id = pair_key.split(":")[0]
            user_id = pair_key.split(":")[1] if ":" in pair_key else "unknown"
            
            # High latency per bot
            if metrics["avg_latency_ms"] > 8000:
                suggestions.append({
                    "id": f"lat-{bot_id}-{int(time.time())}",
                    "priority": "critical",
                    "category": "performance",
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "title": f"{bot_id} has critical latency",
                    "description": f"{metrics['avg_latency_ms']:.0f}ms average response time. May indicate model loading issues.",
                    "triggering_metrics": {"avg_latency_ms": metrics["avg_latency_ms"]}
                })
            
            # Low engagement (few turns per session)
            if metrics.get("avg_turns_per_session", 0) < 3 and metrics["total_turns"] > 10:
                suggestions.append({
                    "id": f"eng-{bot_id}-{int(time.time())}",
                    "priority": "medium",
                    "category": "engagement",
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "title": f"{bot_id} has low session depth",
                    "description": f"Only {metrics['avg_turns_per_session']:.1f} turns per session. Consider more engaging questions.",
                    "suggested_prompt_adjustments": [
                        "Add more follow-up questions to system prompt",
                        "Include conversational hooks in responses"
                    ]
                })
            
            # High tool usage
            if metrics["avg_tool_calls"] > 3:
                suggestions.append({
                    "id": f"tool-{bot_id}-{int(time.time())}",
                    "priority": "medium",
                    "category": "tools",
                    "bot_id": bot_id,
                    "user_id": user_id,
                    "title": f"{bot_id} over-tooling",
                    "description": f"{metrics['avg_tool_calls']:.1f} tool calls per turn. Consider context consolidation.",
                })
        
        return sorted(suggestions, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["priority"], 4))


# =============================================================================
# Routes
# =============================================================================

@router.get("/dashboard", response_model=EvolutionDashboardSummary)
async def get_evolution_dashboard(
    period_hours: int = Query(24, ge=1, le=168, description="Analysis period")
):
    """Get high-level evolution dashboard summary.
    
    This is the main entry point for the evolution dashboard UI.
    """
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs unavailable")
    
    analytics = EvolutionAnalytics(store)
    analysis = analytics.analyze_turns(since_hours=period_hours)
    
    suggestions = analytics.generate_suggestions(analysis)
    
    # Count by priority
    critical = sum(1 for s in suggestions if s["priority"] == "critical")
    high = sum(1 for s in suggestions if s["priority"] == "high")
    medium = sum(1 for s in suggestions if s["priority"] == "medium")
    
    # Find top/bottom performers
    pair_metrics = analysis.get("pair_metrics", {})
    top_bot = None
    bot_attention = None
    
    if pair_metrics:
        by_turns = sorted(pair_metrics.items(), key=lambda x: x[1]["total_turns"], reverse=True)
        if by_turns:
            top_bot = by_turns[0][0].split(":")[0]
        
        by_latency = sorted(pair_metrics.items(), key=lambda x: x[1]["avg_latency_ms"], reverse=True)
        if by_latency and by_latency[0][1]["avg_latency_ms"] > 5000:
            bot_attention = by_latency[0][0].split(":")[0]
    
    # Convert suggestions to schema
    suggestion_models = [
        ImprovementSuggestion(
            id=s["id"],
            priority=s["priority"],
            category=s["category"],
            bot_id=s["bot_id"],
            user_id=s.get("user_id"),
            title=s["title"],
            description=s["description"],
            triggering_metrics=s.get("triggering_metrics"),
            suggested_prompt_adjustments=s.get("suggested_prompt_adjustments"),
            suggested_config_changes=s.get("suggested_config_changes"),
            created_at=datetime.utcnow().isoformat()
        )
        for s in suggestions[:10]  # Top 10
    ]
    
    agg = analysis.get("aggregated", {})
    
    return EvolutionDashboardSummary(
        generated_at=datetime.utcnow().isoformat(),
        period_hours=period_hours,
        total_turns=analysis.get("total_turns", 0),
        total_bots=len(set(k.split(":")[0] for k in pair_metrics.keys())),
        active_bot_user_pairs=analysis.get("unique_pairs", 0),
        avg_system_latency=agg.get("system_avg_latency", 0),
        error_rate=agg.get("system_error_rate", 0),
        top_performing_bot=top_bot,
        bot_needing_attention=bot_attention,
        critical_suggestions=critical,
        high_suggestions=high,
        medium_suggestions=medium,
        recent_suggestions=suggestion_models
    )


@router.get("/bots/{bot_id}/metrics", response_model=BotEvolutionMetrics)
async def get_bot_evolution_metrics(
    bot_id: str,
    period_hours: int = Query(24, ge=1, le=168)
):
    """Get detailed evolution metrics for a specific bot."""
    
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs unavailable")
    
    analytics = EvolutionAnalytics(store)
    analysis = analytics.analyze_turns(bot_id=bot_id, since_hours=period_hours)
    
    if analysis.get("empty"):
        raise HTTPException(status_code=404, detail=f"No data for bot {bot_id}")
    
    # Aggregate across all users for this bot
    pair_metrics = analysis.get("pair_metrics", {})
    bot_pairs = {k: v for k, v in pair_metrics.items() if k.startswith(f"{bot_id}:")}
    
    if not bot_pairs:
        raise HTTPException(status_code=404, detail=f"No data for bot {bot_id}")
    
    # Calculate aggregate metrics
    total_turns = sum(m["total_turns"] for m in bot_pairs.values())
    avg_latency = sum(m["avg_latency_ms"] for m in bot_pairs.values()) / len(bot_pairs)
    p95_latencies = [m["p95_latency_ms"] for m in bot_pairs.values()]
    avg_p95 = sum(p95_latencies) / len(p95_latencies) if p95_latencies else 0
    
    # Get tool usage from turn logs
    rows, _ = store.list_turns(bot_id=bot_id, since_hours=period_hours, limit=1000, offset=0)
    tool_counts: dict[str, int] = defaultdict(int)
    total_tool_calls = 0
    
    for row in rows:
        tools = json.loads(row.tool_calls_json or "[]")
        for tool in tools:
            name = tool.get("name", "unknown")
            tool_counts[name] += 1
            total_tool_calls += 1
    
    top_tools = [
        {"name": name, "count": count}
        for name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    
    return BotEvolutionMetrics(
        bot_id=bot_id,
        period_hours=period_hours,
        total_turns=total_turns,
        unique_users=len(bot_pairs),
        total_messages=total_turns * 2,  # Approximate
        avg_latency_ms=avg_latency,
        p95_latency_ms=avg_p95,
        streaming_usage_rate=sum(m["streaming_rate"] for m in bot_pairs.values()) / len(bot_pairs),
        avg_turns_per_session=sum(m.get("avg_turns_per_session", 0) for m in bot_pairs.values()) / len(bot_pairs),
        user_return_rate=0.5,  # Would need session tracking
        total_tool_calls=total_tool_calls,
        tool_calls_per_turn=total_tool_calls / total_turns if total_turns else 0,
        top_tools=top_tools,
        avg_response_length=int(sum(m["avg_response_length"] for m in bot_pairs.values()) / len(bot_pairs)),
        question_asking_rate=0.0,  # Would need content analysis
        error_rate=sum(m["error_rate"] for m in bot_pairs.values()) / len(bot_pairs),
        user_satisfaction_proxy=0.7,  # Placeholder
    )


@router.get("/bots/{bot_id}/drift", response_model=PersonalityDriftReport)
async def get_bot_personality_drift(
    bot_id: str,
    baseline_days: int = Query(7, ge=1, le=30),
    current_days: int = Query(1, ge=1, le=7)
):
    """Detect personality drift for a bot."""
    
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs unavailable")
    
    analytics = EvolutionAnalytics(store)
    drift = analytics.detect_personality_drift(bot_id, baseline_days, current_days)
    
    return PersonalityDriftReport(
        bot_id=bot_id,
        baseline_period_days=baseline_days,
        comparison_period_days=current_days,
        drift_detected=drift.get("drift_detected", False),
        drift_magnitude=drift.get("drift_magnitude", 0),
        confidence=drift.get("confidence", 0),
        specific_changes=drift.get("changes", []),
        baseline_metrics=drift.get("baseline_metrics", {}),
        current_metrics=drift.get("current_metrics", {}),
        recommended_adjustments=drift.get("recommended_adjustments", [])
    )


@router.get("/suggestions")
async def list_improvement_suggestions(
    bot_id: str | None = Query(None),
    priority: str | None = Query(None, pattern="^(critical|high|medium|low)$"),
    category: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """List improvement suggestions with filtering."""
    
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs unavailable")
    
    analytics = EvolutionAnalytics(store)
    analysis = analytics.analyze_turns(bot_id=bot_id, since_hours=168)  # 1 week
    suggestions = analytics.generate_suggestions(analysis)
    
    # Apply filters
    if bot_id:
        suggestions = [s for s in suggestions if s["bot_id"] == bot_id or s["bot_id"] == "system"]
    if priority:
        suggestions = [s for s in suggestions if s["priority"] == priority]
    if category:
        suggestions = [s for s in suggestions if s["category"] == category]
    
    return {
        "suggestions": suggestions[:limit],
        "total": len(suggestions),
        "filters": {"bot_id": bot_id, "priority": priority, "category": category}
    }


@router.get("/pairs/{bot_id}/{user_id}/quality", response_model=ConversationQualityScore)
async def get_conversation_quality(
    bot_id: str,
    user_id: str,
    period_hours: int = Query(24, ge=1, le=168)
):
    """Get conversation quality score for a specific bot-user pair."""
    
    service = get_service()
    store = TurnLogStore(service.config, ttl_hours=168)
    
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs unavailable")
    
    rows, total = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        since_hours=period_hours,
        limit=1000,
        offset=0
    )
    
    if not rows:
        raise HTTPException(status_code=404, detail="No conversation data found")
    
    # Calculate quality metrics
    turn_count = len(rows)
    
    # Engagement: more turns = higher engagement
    engagement = min(turn_count / 20, 1.0)  # Cap at 20 turns
    
    # Coherence: check for response length consistency
    lengths = [len(r.response_text or "") for r in rows if r.response_text]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        coherence = max(0, 1 - (variance / (avg_len ** 2 + 1)))  # Normalized
    else:
        coherence = 0.5
    
    # Flags
    flags = []
    if turn_count < 3:
        flags.append("shallow_conversation")
    if any(r.error_text for r in rows):
        flags.append("errors_detected")
    
    return ConversationQualityScore(
        bot_id=bot_id,
        user_id=user_id,
        turn_count=turn_count,
        engagement_score=engagement,
        coherence_score=coherence,
        personality_consistency=0.8,  # Would need deeper analysis
        flags=flags,
        signals={
            "avg_response_length": sum(lengths) / len(lengths) if lengths else 0,
            "response_variance": variance if lengths else 0
        }
    )
