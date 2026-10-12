; Jump, double jump, triple jump, and landing on enemies.
;
; The game has no jump, but a leader is a body in its physics: give the body an upward speed and
; gravity does the rest. A jump is the punch state with different answers. The clip it plays is a
; jump's, "look for something to hit" is every frame he is coming down, the place searched is under
; him, and the state ends when he has landed. A punch passes straight through to punch.s.
;
; Names in capitals and the game's own functions come from moves.py. The capitals used as offsets
; from a pointer to "the jump's bytes" (ASKED, IS_JUMP, WHICH, PHASE, FRAMES, WINDOW, GROUND) are
; its eight bytes of memory.

; ==== the walk state, once a frame, where it moves him: r4 the state, r8 the leader, r11 his pad ====
.org room_jump
j_walk:                                 ; stands for the call to the game's steering; r0-r2 are that call's
    push  {r5, lr}
    bl    steer
    ldr   r12, [r8, #0x81c]
    cmp   r12, #MARIO                   ; only the leader whose model is his
    popne {r5, pc}
    ldr   r5, bytes_from_walk
walk_at:
    add   r5, pc, r5                    ; r5 -> the jump's bytes
    ldr   r3, [r8, #0x650]              ; r3: his body
    ldrb  r1, [r5, #PHASE]
    cmp   r1, #0                        ; a jump he was knocked out of, or one just over: put things back
    movne r1, #0
    strbne r1, [r5, #PHASE]
    strbne r1, [r3, #0x2f6]             ; (the body holds to the ground again)
    ldrb  r1, [r5, #WINDOW]
    subs  r1, r1, #1
    strbpl r1, [r5, #WINDOW]            ; one frame less in which a press is the next jump up
    mov   r0, r11
    bl    pressed
    ldrb  r1, [r3, #0x2cd]
    ldrb  r2, [r3, #0x2d4]
    orrs  r1, r1, r2                    ; only from the ground
    moveq r0, #0
    cmp   r0, #0
    movne r0, #1
    strb  r0, [r5, #ASKED]              ; a jump is asked for, or is not: the punch state reads this on entering
    popeq {r5, pc}
    sub   sp, sp, #8                    ; into the punch state, as the walk state goes into it itself
    mov   r0, sp
    mov   r1, r8
    bl    hold
    ldr   r5, [r4, #0x3c]               ; the leader's state machine
    ldr   r0, [r5, #0x44]
    cmp   r0, #0
    movne r0, r5
    blne  drop_pending
    mov   r0, r5
    mov   r1, #0x30
    bl    make_punch
    str   r0, [r5, #0x44]
    str   r5, [r0, #0x48]
    mov   r1, #0
    strh  r1, [r5, #0x58]
    mov   r0, sp
    bl    release
    add   sp, sp, #8
    pop   {r5, lr}
    b     walk_return                   ; and the walk state is done for this frame, as after its own changes of state
bytes_from_walk: .word JUMP_STATE - (walk_at + 8)

; ---- was the jump button pressed this frame? r0 a pad; r0 nonzero if so. Changes r1 and nothing else ----
pressed:
    ldr   r1, [r0, #0x238]              ; the controller behind the pad
    ldrb  r0, [r0, #9]
    and   r0, r0, #8                    ; a click of the right stick
    cmp   r1, #0
    ldrbne r1, [r1, #0xa]
    and   r1, r1, #4                    ; down on the D-pad, which the game's table maps to nothing
    orr   r0, r0, r1
    bx    lr

; ==== entering the punch state: r0 the animation, r2 the body part, r3 the play function, r4 the state ====
j_enter_a:
    ldr   r12, [r4, #0x5c]              ; the leader
    b     j_enter
j_enter_b:
    ldr   r12, [r4, #0x28]              ; the same, from the twin's side of the state
j_enter:
    ldr   r1, [r12, #0x81c]
    cmp   r1, #MARIO
    bne   enter                         ; anyone else: the punches' code, which says the game's own clip
    push  {r0, r3, r7, lr}
    mov   r7, r12
    ldr   r12, bytes_from_enter
j_enter_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #ASKED]             ; was a jump asked for?
    strb  r1, [r12, #IS_JUMP]           ; then this is one
    cmp   r1, #0
    bne   jump_enter
    mov   r12, r7                       ; no: a punch, as it was
    pop   {r0, r3, r7, lr}
    b     enter
jump_enter:
    ldrb  r1, [r12, #PHASE]
    cmp   r1, #0                        ; (the state has two ways in; should the game use both, the second finds him
    bne   entered                       ;  in the air already and only says the clip again)
    bl    next_jump
    bl    lift
entered:
    ldrb  r1, [r12, #WHICH]
    add   r1, r1, #JUMP_CLIP - 1
    mov   r2, #2                        ; on the whole body
    pop   {r0, r3, r7, pc}
bytes_from_enter: .word JUMP_STATE - (j_enter_at + 8)

; ---- which of the three: r12 the jump's bytes, r7 the leader. Changes r1-r3 ----
next_jump:
    ldrb  r1, [r12, #WHICH]             ; the one before
    ldrb  r2, [r12, #WINDOW]
    cmp   r2, #0
    moveq r1, #0                        ; too long ago: the first again
    cmp   r1, #2
    bne   count_up
    ldr   r2, [r7, #0x828]              ; the third needs a run-up: his speed
    ldr   r3, run_up
    cmp   r2, r3                        ; (a speed is never negative, so its bits compare as it does)
    movcc r1, #0
count_up:
    cmp   r1, #3
    movcs r1, #0                        ; after the third, the first
    add   r1, r1, #1
    strb  r1, [r12, #WHICH]
    bx    lr
run_up: .word RUN_UP

; ---- leave the ground: r12 the jump's bytes, r7 the leader. Changes r1-r3; r1 = the jump's clip ----
lift:
    mov   r1, #1
    strb  r1, [r12, #PHASE]             ; in the air
    mov   r2, #0
    strh  r2, [r12, #FRAMES]
    strb  r2, [r12, #WINDOW]
    ldr   r2, [r7, #0x650]              ; his body
    strb  r1, [r2, #0x2f6]              ; it lets go of the ground (the byte the game's own falls set)
    ldrb  r1, [r12, #WHICH]
    add   r3, pc, r1, lsl #2
    ldr   r3, [r3, #4]                  ; the speed of this jump, from the three below
    str   r3, [r2, #0x120]              ; upward
    b     lift_clip
    .word SPEED_1
    .word SPEED_2
    .word SPEED_3
lift_clip:
    add   r1, r1, #JUMP_CLIP - 1
    bx    lr

; ==== each frame of the punch state, "has the clip ended?": r0 the animation, r4 the state, r7 the leader ====
j_update:
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    bne   update
    ldr   r12, bytes_from_update
j_update_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #IS_JUMP]
    cmp   r1, #0
    beq   update                        ; a punch: the punches' code
    push  {r4, r5, r6, r8, r9, lr}
    mov   r6, r12                       ; r6 -> the jump's bytes
    bl    anim_flags
    mov   r5, r0                        ; r5: the answer
    ldr   r8, [r7, #0x650]              ; r8: his body
    mov   r0, r7
    bl    pad_of
    mov   r9, r0                        ; r9: the pad of whoever plays him, or 0
    ldrb  r0, [r6, #PHASE]
    cmp   r0, #1
    bne   landed
    bic   r5, r5, #0x80000000           ; in the air: the clip running out does not end the jump
    ldrh  r0, [r6, #FRAMES]
    add   r0, r0, #1
    strh  r0, [r6, #FRAMES]
    cmp   r0, #SETTLE
    bcc   in_air                        ; the ground he has just left does not count
    ldrb  r1, [r8, #0x2cd]
    ldrb  r2, [r8, #0x2d4]
    orrs  r1, r1, r2
    bne   touch_down
    cmp   r0, #LONGEST
    bcs   give_up                       ; never coming down: let him go
in_air:
    cmp   r9, #0
    beq   j_done
    ldr   r0, [r9, #0x210]              ; the stick, across and along
    ldr   r1, [r9, #0x214]
    orr   r0, r0, r1
    bics  r0, r0, #0x80000000
    beq   j_done                        ; let go: he keeps the speed he has
    ldrb  r0, [r8, #0x2cd]
    strb  r0, [r6, #GROUND]
    mov   r0, #1
    strb  r0, [r8, #0x2cd]              ; the game's steering steers only a body on the ground: say so for the call
    bl    steer_him
    ldrb  r0, [r6, #GROUND]
    strb  r0, [r8, #0x2cd]
    b     j_done
touch_down:
    mov   r0, #0
    strb  r0, [r8, #0x2f6]              ; the body holds to the ground again
    strh  r0, [r6, #FRAMES]
    mov   r0, #2
    strb  r0, [r6, #PHASE]              ; landing
    mov   r0, #CHAIN
    strb  r0, [r6, #WINDOW]             ; a press in this many frames is the next jump up
    ldrb  r1, [r6, #WHICH]
    add   r1, r1, #LAND_CLIP - 1
    bl    play_whole                    ; the landing of this jump
    b     j_done
landed:
    bl    steer_him                     ; he can run on through a landing
    ldrh  r0, [r6, #FRAMES]
    add   r0, r0, #1
    strh  r0, [r6, #FRAMES]
    ldrb  r1, [r6, #WINDOW]
    subs  r1, r1, #1
    strbpl r1, [r6, #WINDOW]
    cmp   r9, #0
    beq   j_done                        ; nobody playing him: the landing plays out
    mov   r0, r9
    bl    pressed
    cmp   r0, #0
    bne   again
    ldrh  r0, [r6, #FRAMES]
    cmp   r0, #RUN_ON
    bcc   j_done
    ldr   r0, [r9, #0x210]
    ldr   r1, [r9, #0x214]
    orr   r0, r0, r1
    bics  r0, r0, #0x80000000
    orrne r5, r5, #0x80000000           ; the stick is over: the landing is cut short and he walks
    b     j_done
again:
    mov   r12, r6
    bl    next_jump
    bl    lift
    bl    play_whole
    bic   r5, r5, #0x80000000
    b     j_done
give_up:
    mov   r0, #0
    strb  r0, [r8, #0x2f6]
    strb  r0, [r6, #PHASE]
    orr   r5, r5, #0x80000000
j_done:
    mov   r0, r5
    pop   {r4, r5, r6, r8, r9, pc}
bytes_from_update: .word JUMP_STATE - (j_update_at + 8)

; ---- play a clip on his whole body: r1 the clip, r7 the leader ----
play_whole:
    ldr   r0, [r7, #0x374]              ; his animation
    ldr   r12, [r0]
    mov   r2, #2
    ldr   r12, [r12, #0xc4]
    bx    r12
; ---- the walk state's steering, for him, from this state: r4 the state, r7 the leader ----
steer_him:
    mov   r0, r4
    mov   r1, r7
    mov   r2, #0
    b     steer

; ---- the pad of whoever plays a leader, found as the walk state finds it: r0 the leader; r0 the pad, or 0 ----
pad_of:
    push  {r4, lr}
    bl    player_of
    cmn   r0, #1
    moveq r0, #0
    popeq {r4, pc}                      ; nobody is
    mov   r4, r0
    ldr   r0, mode_from_pad
pad_mode_at:
    ldr   r0, [pc, r0]
    cmp   r0, #0
    beq   pad_have
    bl    shared_mode
    cmp   r0, #0
    beq   pad_have
    ldr   r0, game_from_pad
pad_game_at:
    ldr   r0, [pc, r0]
    bl    shared_pad
    mov   r4, r0
pad_have:
    ldr   r0, pads_from_pad
pad_pads_at:
    ldr   r0, [pc, r0]
    add   r0, r0, #0x10
    cmp   r4, #4
    addcc r0, r0, r4, lsl #10           ; + player * 0x4f0
    addcc r0, r0, r4, lsl #8
    subcc r0, r0, r4, lsl #4
    pop   {r4, pc}
mode_from_pad: .word mode_object - (pad_mode_at + 8)
game_from_pad: .word game_object - (pad_game_at + 8)
pads_from_pad: .word pads - (pad_pads_at + 8)

; ==== "is this the frame it lands?", asked before the game looks for a target: r0 the animation, r7 the leader ====
j_flags:
    push  {r4, lr}
    bl    anim_flags
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    popne {r4, pc}
    ldr   r12, bytes_from_flags
flags_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #IS_JUMP]
    cmp   r1, #0
    popeq {r4, pc}                      ; a punch: its clip's own marker
    bic   r0, r0, #0x400                ; a jump lands on no frame of its clip ...
    ldrb  r1, [r12, #PHASE]
    cmp   r1, #1
    popne {r4, pc}
    ldr   r1, [r7, #0x650]
    ldr   r1, [r1, #0x120]              ; ... but on every frame he is in the air and coming down
    cmp   r1, #0
    orrmi r0, r0, #0x400
    pop   {r4, pc}
bytes_from_flags: .word JUMP_STATE - (flags_at + 8)

; ==== the numbers of a hit, each where the game set one: r7 the leader. A punch's are the punches' code's ====
j_reach:                                ; r0: the radius searched; s18 (already 4.0): how far ahead of him its centre is
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    bne   reach
    ldr   r12, bytes_from_reach
reach_at:
    add   r12, pc, r12
    ldrb  r12, [r12, #IS_JUMP]
    cmp   r12, #0
    beq   reach
    ldr   r0, stomp_radius
    mov   r12, #0                       ; under him, not ahead
    vmov  s18, r12
    bx    lr
j_furthest:                             ; s16: the furthest from him a hit may land; r0 is the game's, kept
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    bne   furthest
    ldr   r12, bytes_from_furthest
furthest_at:
    add   r12, pc, r12
    ldrb  r12, [r12, #IS_JUMP]
    cmp   r12, #0
    beq   furthest
    ldr   r12, stomp_furthest
    vmov  s16, r12
    bx    lr
j_strength:                             ; r0: the strength of the hit
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    bne   strength
    ldr   r12, bytes_from_strength
strength_at:
    add   r12, pc, r12
    ldrb  r12, [r12, #IS_JUMP]
    cmp   r12, #0
    beq   strength
    ldr   r0, stomp_power
    bx    lr
bytes_from_reach:    .word JUMP_STATE - (reach_at + 8)
bytes_from_furthest: .word JUMP_STATE - (furthest_at + 8)
bytes_from_strength: .word JUMP_STATE - (strength_at + 8)
stomp_radius:   .word STOMP_RADIUS
stomp_furthest: .word STOMP_FURTHEST
stomp_power:    .word STOMP_POWER

; ==== handing the enemy the hit: r0 the enemy, r1 the hit, r2 the enemy's function for it, r7 the leader ====
j_send:
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    bxne  r2                            ; anyone else: the game's own call
    ldr   r12, bytes_from_send
send_at:
    add   r12, pc, r12
    ldrb  r12, [r12, #IS_JUMP]
    cmp   r12, #0
    bxeq  r2                            ; a punch: the same
    push  {r4, r5, r6, r8, r9, lr}
    mov   r4, r0
    mov   r5, r1
    mov   r6, r2
    ldr   r8, [r4, #0x3a8]              ; the enemy's life now
    mov   r3, #1
    str   r3, [r5, #8]                  ; first as the game's "landed on from above": the kind of hit a thrown
    strb  r3, [r5, #0x14]               ; Pikmin's landing is, with its mark, which kills what can be squashed
    blx   r6
    ldr   r3, [r4, #0x3a8]
    cmp   r3, r8
    bne   bounce
    mov   r3, #0                        ; its life has not moved: it cannot be squashed
    str   r3, [r5, #8]
    strb  r3, [r5, #0x14]
    mov   r0, r4                        ; as a blow, with the stomp's strength
    mov   r1, r5
    blx   r6
    ldr   r3, [r4, #0x3a8]
    cmp   r3, r8
    popeq {r4, r5, r6, r8, r9, pc}      ; nothing came of it: no bounce
bounce:
    ldr   r1, [r7, #0x650]
    ldr   r2, bounce_speed
    str   r2, [r1, #0x120]              ; and up off it
    pop   {r4, r5, r6, r8, r9, pc}
bytes_from_send: .word JUMP_STATE - (send_at + 8)
bounce_speed:    .word BOUNCE
