#!/bin/bash
# Code Execution Observability Dashboard
# Usage: ./scripts/code-exec-dashboard.sh [logfile]
# Default logfile: /tmp/agent-stderr.log

LOGFILE="${1:-/tmp/agent-stderr.log}"
NS="${2:-ap-default-agent}"

echo "================================================================"
echo "          CODE EXECUTION OBSERVABILITY DASHBOARD"
echo "================================================================"
echo "  Log: $LOGFILE"
echo "  Namespace: $NS"
echo ""

echo "📈 METRICS (execution stats)"
echo "────────────────────────────"
grep "code_execution_metric" "$LOGFILE" 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        ts = d.get('timestamp', '')[-12:]
        print(f'  [{ts}] lang={d.get(\"language\"):8s} exit={d.get(\"exit_code\")} status={d.get(\"status\"):8s} duration={d.get(\"duration_seconds\")}s')
    except: pass
" || echo "  No executions yet"

echo ""
echo "💰 COST TRACKING (resource usage)"
echo "──────────────────────────────────"
grep "resource_usage" "$LOGFILE" 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        print(f'  cpu={d.get(\"cpu_seconds\",0):.3f}s  memory={d.get(\"memory_mb_seconds\",0):.1f}MB·s  duration={d.get(\"duration_seconds\",0):.1f}s  org={d.get(\"org\")}')
    except: pass
" || echo "  Cost tracking disabled"

echo ""
echo "⏱️  QUEUE STATS (concurrency)"
echo "─────────────────────────────"
grep "queue_wait" "$LOGFILE" 2>/dev/null | python3 -c "
import sys, json
waits = []
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        w = d.get('wait_seconds', 0)
        waits.append(w)
    except: pass
if waits:
    print(f'  Total queued: {len(waits)}')
    print(f'  Avg wait: {sum(waits)/len(waits):.3f}s')
    print(f'  Max wait: {max(waits):.3f}s')
    print(f'  Zero-wait (instant): {sum(1 for w in waits if w == 0)}/{len(waits)}')
else:
    print('  No queue data')
"
rejected=$(grep -c "code_execution_rejected" "$LOGFILE" 2>/dev/null | tr -d "\n")
echo "  Rejected (queue full): $rejected"

echo ""
echo "🔒 NETWORK POLICIES (per-execution)"
echo "─────────────────────────────────────"
np_created=$(grep -c "network_policy_created" "$LOGFILE" 2>/dev/null | tr -d "\n")
np_deleted=$(grep -c "network_policy_deleted" "$LOGFILE" 2>/dev/null | tr -d "\n")
np_leaked=$((np_created - np_deleted))
echo "  Created: $np_created  Deleted: $np_deleted  Leaked: $np_leaked"

echo ""
echo "📁 FILE I/O (ConfigMaps)"
echo "────────────────────────"
cm_created=$(grep -c "configmap_created" "$LOGFILE" 2>/dev/null | tr -d "\n")
cm_deleted=$(grep -c "configmap_deleted" "$LOGFILE" 2>/dev/null | tr -d "\n")
cm_leaked=$((cm_created - cm_deleted))
echo "  Created: $cm_created  Deleted: $cm_deleted  Leaked: $cm_leaked"

echo ""
echo "🎬 STREAMING"
echo "─────────────"
grep "streaming_started\|streaming_completed" "$LOGFILE" 2>/dev/null | python3 -c "
import sys, json
started = 0; completed = 0; total_bytes = 0
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        evt = d.get('event', '')
        if 'started' in evt: started += 1
        elif 'completed' in evt:
            completed += 1
            total_bytes += d.get('total_bytes', 0)
    except: pass
if started:
    print(f'  Sessions: {started} started, {completed} completed')
    print(f'  Total bytes streamed: {total_bytes}')
else:
    print('  No streaming events (streaming_enabled: false)')
"

echo ""
echo "❌ ERRORS"
echo "─────────"
grep "code_execution_failed\|code_execution_error_metric\|cleanup_failed\|network_policy_delete_failed\|configmap_delete_failed" "$LOGFILE" 2>/dev/null | python3 -c "
import sys, json
errors = []
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        ts = d.get('timestamp', '')[-12:]
        evt = d.get('event', '')
        err = d.get('error', d.get('error_type', ''))
        job = d.get('job_name', '')
        errors.append(f'  [{ts}] {evt}: {err} (job={job})')
    except: pass
if errors:
    for e in errors[-5:]: print(e)
    if len(errors) > 5: print(f'  ... and {len(errors)-5} more')
else:
    print('  No errors')
"

echo ""
echo "☸️  K8s RESOURCES (should be empty = all cleaned up)"
echo "────────────────────────────────────────────────────"
resources=$(kubectl get jobs,pods,networkpolicies -n "$NS" --no-headers 2>/dev/null | grep -v "kube-root")
if [ -n "$resources" ]; then
    echo "  ⚠️  LEAKED RESOURCES:"
    echo "$resources" | sed 's/^/  /'
else
    echo "  ✅ All clean — no leaked resources"
fi

echo ""
echo "================================================================"
total=$(grep -c 'code_execution_completed' "$LOGFILE" 2>/dev/null || true)
success=$(grep 'code_execution_metric' "$LOGFILE" 2>/dev/null | grep -c '"status": "success"' || true)
failed=$((total - success))
echo "  Total: $total executions | Success: $success | Failed: $failed"
echo "  Network policies: $np_created created, $np_deleted deleted, $np_leaked leaked"
echo "  ConfigMaps: $cm_created created, $cm_deleted deleted, $cm_leaked leaked"
echo "================================================================"
