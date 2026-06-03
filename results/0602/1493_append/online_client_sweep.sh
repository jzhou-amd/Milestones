#!/bin/bash

> /vllm-workspace/output/request_timeline.json 2>/dev/null || true

# Three distinct prefixes
PREFIX_A=$(printf 'SCENARIO_A: Once upon a time in a land far away, there lived a wise old wizard who knew many secrets of the ancient world. He wandered through forests and mountains, discovering magical artifacts and ancient spells. %.0s' {1..700})

PREFIX_B=$(printf 'SCENARIO_B: In a distant galaxy beyond the stars, a brave space explorer navigated through asteroid fields and nebulas, searching for signs of alien civilizations and lost technologies from forgotten eras. %.0s' {1..700})

PREFIX_C=$(printf 'SCENARIO_C: Deep beneath the ocean waves, a marine biologist discovered bioluminescent creatures and ancient underwater ruins, revealing mysteries of prehistoric aquatic civilizations and their advanced technologies. %.0s' {1..700})

APPENDAGE=$(printf 'Youre an animal, Sibling Dex. You are not separate or other. Youre an animal. And animals have no purpose. Nothing has a purpose. The world simply is. If you want to do things that are meaningful to others, fine! Good! So do I! But if I wanted to crawl into a cave and watch stalagmites with Frostfrog for the remainder of my days, that would also be both fine and good. You keep asking why your work is not enough, and I dont know how to answer that, because it is enough to exist in the world and marvel at it. You dont need to justify that, or earn it. You are allowed to just live. That is all most animals do. %.0s' {1..10})

echo "PREFIX_A length: ${#PREFIX_A} chars"
echo "PREFIX_B length: ${#PREFIX_B} chars"
echo "PREFIX_C length: ${#PREFIX_C} chars"
echo ""

send_request() {
  local prefix="$1"
  local question="$2"
  local req_num="$3"
  
  echo "Request $req_num: Sending..."
  
  printf '{"model":"Qwen/Qwen2-7B-Instruct","prompt":"%s%s","max_tokens":1,"temperature":0.0}' \
    "$prefix" "$question" | \
  curl -s http://localhost:8001/v1/completions \
    -H 'Content-Type: application/json' \
    -d @- > /dev/null &
}

echo "=== PHASE 1: Cache PREFIX_A (5 requests) ==="
for i in {1..2}; do
  send_request "$PREFIX_A" "Question $i: What is the meaning of life?" "$i"
  sleep 0.2
done

echo "=== PHASE 2: Cache PREFIX_B (10 requests) ==="
for i in {1..2}; do
  send_request "$PREFIX_B" "Question $i: What is the nature of reality?" "$((i+5))"
  sleep 0.2
done

echo "=== PHASE 3: Cache PREFIX_C (10 requests) - should evict A ==="
for i in {1..2}; do
  send_request "$PREFIX_C" "Question $i: What is the purpose of existence?" "$((i+15))"
  sleep 0.2
done

echo "=== PHASE 4: Return to PREFIX_A (10 requests) - should LOAD from CPU ==="
for i in {1..2}; do
  send_request "${PREFIX_A}${APPENDAGE}" "Question $((i+10)): What is consciousness?" "$((i+25))"
  sleep 0.2
done

wait

echo ""
echo "=== All requests complete! ==="
echo "Now check for EVICT_START and LOAD_START events in the profiler output!"
