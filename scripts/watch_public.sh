#!/usr/bin/env bash
# Watch the public endpoint and the JevBench bench request while Benchmark Heaven measures it.
#
#   nohup bash scripts/watch_public.sh > /dev/null 2>&1 &   # start the loop (one poll every 5 min)
#   bash scripts/watch_public.sh --status                   # one-shot summary
#   bash scripts/watch_public.sh --once                     # one poll cycle, then exit
#   pkill -f 'watch_publi[c]\.sh$'                          # stop the loop
#
# Events are appended to results/public/watch_public.log as "<ts> <KIND> <detail>":
#   WATCHER START/STOP, COMMENT (new comment on the issue), STATE (issue opened/closed),
#   TUNNEL DOWN/RECOVERED/RESTARTED/STALE, SERVER DOWN/RECOVERED/RESTARTED/HUNG,
#   CAFFEINATE RESTARTED, TRAFFIC BASELINE/START/END (bursts of external requests through the
#   tunnel, from cloudflared's local /metrics counters), REQUESTS (external requests per cycle),
#   HB (heartbeat every 6 h).
# A process that has disappeared is restarted with the commands from README "公開 endpoint".
# A restarted Quick Tunnel gets a NEW url: the issue must then be updated by hand
# (this script never writes to GitHub).
set -u
cd "$(dirname "$0")/.."

REPO=${JQV_WATCH_REPO:-fstandhartinger/jevbench}
ISSUE=${JQV_WATCH_ISSUE:-6}
INTERVAL=${JQV_WATCH_INTERVAL:-300}
HB_EVERY=${JQV_WATCH_HEARTBEAT:-21600}
PUB=results/public
LOG=$PUB/watch_public.log
SEEN=$PUB/watch_seen_comments.txt
STATE_FILE=$PUB/watch_issue_state.txt
TRAFFIC_FILE=$PUB/watch_traffic.txt
URL_FILE=$PUB/tunnel_url.txt
TEMP_FILE=results/mmlu_packed_qwen3-32b_temperature.json

tfail=0; lfail=0; tunnel_bad=0; server_bad=0; last_hb=0
url=""; code=000; local_code=000; own_requests=0
m_total=0; m_2xx=0; m_4xx=0; m_5xx=0; active=0; idle=0; ext_sum=0; act_cycles=0

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }
http() { curl -s -m "${2:-20}" -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo 000; }
pub_url() { tr -d '[:space:]' < "$URL_FILE" 2>/dev/null; }
alive_uvicorn() { pgrep -f "uvicorn jqv.server:app" >/dev/null; }
alive_tunnel() { pgrep -f "cloudflared tunnel --url" >/dev/null; }
alive_caffeinate() { pgrep -x caffeinate >/dev/null; }

restart_uvicorn() {
  JQV_MODEL=Qwen/Qwen3-32B JQV_ENGINE=packed JQV_TEMPERATURE_FILE=$TEMP_FILE \
    nohup uv run uvicorn jqv.server:app --host 127.0.0.1 --port 8000 >> "$PUB/server.log" 2>&1 &
  log "SERVER RESTARTED pid=$! (the 32B load takes a few minutes; local health follows)"
}

restart_tunnel() {
  local before pid new=""
  before=$(wc -l < "$PUB/cloudflared.log" 2>/dev/null || echo 0)
  nohup cloudflared tunnel --url http://localhost:8000 >> "$PUB/cloudflared.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 30); do
    sleep 2
    new=$(tail -n +"$((before + 1))" "$PUB/cloudflared.log" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
    [ -n "$new" ] && break
  done
  if [ -n "$new" ]; then
    printf '%s\n' "$new" > "$URL_FILE"
    log "TUNNEL RESTARTED pid=$pid new_url=$new (issue #$ISSUE still names the old url; update it by hand)"
  else
    log "TUNNEL RESTART FAILED pid=$pid (no url in cloudflared.log after 60 s)"
  fi
}

restart_caffeinate() { nohup caffeinate -dimsu >/dev/null 2>&1 & log "CAFFEINATE RESTARTED pid=$!"; }

check_issue() {
  local out f1 f2 f3 f4 prev
  out=$(gh issue view "$ISSUE" -R "$REPO" --json state,comments \
        --jq '"STATE\t\(.state)", (.comments[] | [.url, .createdAt, .author.login, (.body | gsub("[\\r\\n\\t]+"; " ") | .[:600])] | @tsv)' 2>/dev/null) || return 0
  [ -n "$out" ] || return 0
  touch "$SEEN"
  while IFS=$'\t' read -r f1 f2 f3 f4; do
    if [ "$f1" = STATE ]; then
      prev=$(cat "$STATE_FILE" 2>/dev/null || true)
      if [ "$prev" != "$f2" ]; then
        printf '%s\n' "$f2" > "$STATE_FILE"
        [ -n "$prev" ] && log "STATE issue #$ISSUE is now $f2 (was $prev)"
      fi
    elif [ -n "$f1" ] && ! grep -qxF "$f1" "$SEEN"; then
      printf '%s\n' "$f1" >> "$SEEN"
      log "COMMENT $f2 $f3: $f4 ($f1)"
    fi
  done <<< "$out"
}

check_endpoint() {
  url=$(pub_url)
  code=$(http "$url/health" 20)
  [ -n "$url" ] && own_requests=$((own_requests + 1))
  local_code=$(http "http://127.0.0.1:8000/health" 10)

  if [ "$local_code" = 200 ]; then
    [ "$server_bad" = 1 ] && log "SERVER RECOVERED local=200"
    server_bad=0; lfail=0
  else
    lfail=$((lfail + 1))
    if ! alive_uvicorn; then
      log "SERVER DOWN process absent (local http=$local_code); restarting"
      server_bad=1; restart_uvicorn; lfail=-4
    elif [ "$lfail" -eq 2 ]; then
      log "SERVER DOWN local http=$local_code, process alive (2 consecutive checks)"; server_bad=1
    elif [ "$lfail" -eq 6 ]; then
      log "SERVER HUNG local http=$local_code for 6 checks, process alive (not restarted automatically)"; server_bad=1
    fi
  fi

  if [ "$code" = 200 ]; then
    [ "$tunnel_bad" = 1 ] && log "TUNNEL RECOVERED $url http=200"
    tunnel_bad=0; tfail=0
  else
    tfail=$((tfail + 1))
    if ! alive_tunnel; then
      log "TUNNEL DOWN process absent (http=$code); restarting"
      tunnel_bad=1; restart_tunnel; tfail=-2
    elif [ "$local_code" != 200 ]; then
      :  # server-side problem, reported above
    elif [ "$tfail" -eq 2 ]; then
      log "TUNNEL DOWN $url http=$code, cloudflared alive, local=200 (2 consecutive checks)"; tunnel_bad=1
    elif [ "$tfail" -eq 6 ]; then
      log "TUNNEL STALE $url http=$code for 6 checks, cloudflared alive (not restarted automatically: a restart changes the url)"; tunnel_bad=1
    fi
  fi

  alive_caffeinate || restart_caffeinate
}

# cloudflared serves Prometheus counters on a local port; they count every request through the tunnel.
read_metrics() {
  local port m
  port=$(lsof -nP -iTCP -a -c cloudflared 2>/dev/null | awk '/LISTEN/ {sub(".*:", "", $9); print $9; exit}')
  [ -n "$port" ] || return 1
  m=$(curl -s -m 5 "http://127.0.0.1:$port/metrics" 2>/dev/null) || return 1
  [ -n "$m" ] || return 1
  m_total=$(printf '%s\n' "$m" | awk '$1 == "cloudflared_tunnel_total_requests" {print int($2)}')
  m_2xx=$(printf '%s\n' "$m" | awk -F'[{}" =]+' '$1 == "cloudflared_tunnel_response_by_code" && $3 ~ /^2/ {s += $4} END {print s + 0}')
  m_4xx=$(printf '%s\n' "$m" | awk -F'[{}" =]+' '$1 == "cloudflared_tunnel_response_by_code" && $3 ~ /^4/ {s += $4} END {print s + 0}')
  m_5xx=$(printf '%s\n' "$m" | awk -F'[{}" =]+' '$1 == "cloudflared_tunnel_response_by_code" && $3 ~ /^5/ {s += $4} END {print s + 0}')
  [ -n "$m_total" ]
}

check_traffic() {
  read_metrics || return 0
  local p_total p2 p4 p5 d d2 d4 d5 ext
  if [ ! -s "$TRAFFIC_FILE" ]; then
    printf '%s %s %s %s\n' "$m_total" "$m_2xx" "$m_4xx" "$m_5xx" > "$TRAFFIC_FILE"
    log "TRAFFIC BASELINE total=$m_total 2xx=$m_2xx 4xx=$m_4xx 5xx=$m_5xx (cloudflared counters since its start)"
    own_requests=0
    return 0
  fi
  read -r p_total p2 p4 p5 < "$TRAFFIC_FILE"
  printf '%s %s %s %s\n' "$m_total" "$m_2xx" "$m_4xx" "$m_5xx" > "$TRAFFIC_FILE"
  d=$((m_total - p_total)); d2=$((m_2xx - p2)); d4=$((m_4xx - p4)); d5=$((m_5xx - p5))
  ext=$((d - own_requests)); ext2=$((d2 - own_requests)); own_requests=0
  [ "$ext2" -lt 0 ] && ext2=0
  if [ "$d" -lt 0 ]; then log "TRAFFIC COUNTER RESET total=$m_total (cloudflared restarted?)"; return 0; fi
  if [ "$ext" -gt 0 ]; then
    log "REQUESTS +$ext external in ${INTERVAL}s (2xx=+$ext2 4xx=+$d4 5xx=+$d5, total=$m_total)"
  fi
  # a burst = successful external responses; 4xx path probes alone do not count
  if [ "$ext2" -ge 2 ] || { [ "$active" = 1 ] && [ "$ext2" -gt 0 ]; }; then
    if [ "$active" = 0 ]; then
      active=1; ext_sum=0; act_cycles=0
      log "TRAFFIC START +$ext2 successful external requests in ${INTERVAL}s (total=$m_total)"
    fi
    ext_sum=$((ext_sum + ext2)); act_cycles=$((act_cycles + 1)); idle=0
  elif [ "$active" = 1 ]; then
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then
      log "TRAFFIC END $ext_sum successful external requests over $act_cycles active cycles (total=$m_total)"
      active=0
    fi
  fi
}

cycle() {
  local now n
  check_issue
  check_endpoint
  check_traffic
  now=$(date +%s)
  if [ $((now - last_hb)) -ge "$HB_EVERY" ]; then
    n=$(grep -c . "$SEEN" 2>/dev/null); n=${n:-0}
    log "HB tunnel=$url http=$code local=$local_code issue=$(cat "$STATE_FILE" 2>/dev/null) comments=$n requests_total=$m_total"
    last_hb=$now
  fi
}

status() {
  local pids n started
  pids=$(pgrep -f 'watch_publi[c]\.sh$' | tr '\n' ' ')
  if [ -n "$pids" ]; then echo "watcher   : running pid $pids"; else echo "watcher   : NOT RUNNING"; fi
  echo "processes : uvicorn=$(alive_uvicorn && echo ok || echo ABSENT) cloudflared=$(alive_tunnel && echo ok || echo ABSENT) caffeinate=$(alive_caffeinate && echo ok || echo ABSENT)"
  url=$(pub_url)
  echo "endpoint  : $url http=$(http "$url/health" 20) local=$(http http://127.0.0.1:8000/health 10)"
  echo "issue     : $(gh issue view "$ISSUE" -R "$REPO" --json state,comments,url --jq '"\(.url) state=\(.state) comments=\(.comments|length)"' 2>/dev/null || echo 'gh unavailable')"
  if read_metrics; then
    started=$(ps -o lstart= -p "$(pgrep -f 'cloudflared tunnel --url' | head -1)" 2>/dev/null | sed 's/  */ /g')
    echo "traffic   : total=$m_total 2xx=$m_2xx 4xx=$m_4xx 5xx=$m_5xx (cloudflared counters since $started)"
  else
    echo "traffic   : cloudflared metrics unavailable"
  fi
  echo "last traffic: $(grep -E ' (REQUESTS|TRAFFIC) ' "$LOG" 2>/dev/null | tail -1)"
  echo "last HB   : $(grep ' HB ' "$LOG" 2>/dev/null | tail -1)"
  n=$(grep -vcE ' (HB|REQUESTS) ' "$LOG" 2>/dev/null)
  echo "events    : ${n:-0} (last 5)"
  grep -vE ' (HB|REQUESTS) ' "$LOG" 2>/dev/null | tail -5
}

case "${1:-loop}" in
  --status) status; exit 0 ;;
  --once) cycle; exit 0 ;;
  loop) ;;
  *) echo "usage: $0 [--status|--once]" >&2; exit 2 ;;
esac

log "WATCHER START pid=$$ interval=${INTERVAL}s issue=$REPO#$ISSUE url=$(pub_url)"
trap 'log "WATCHER STOP pid=$$"; exit 0' TERM INT
while true; do
  cycle
  sleep "$INTERVAL" & wait $!
done
