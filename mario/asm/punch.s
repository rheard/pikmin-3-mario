; Punch, punch, kick.
;
; The leaders' punch state plays one clip and ends with it. For Mario, entering it plays his first
; punch, and a press of A once a clip has passed its "another press is taken from here" marker plays
; the next move and keeps the state alive. The game still does the hitting; the last section hands
; it his numbers in place of a leader's.
;
; Names in capitals and the game's own functions come from moves.py. Every other leader takes the
; first way out of each routine, with the answer the game's own word would have given.

; ---- each frame of the punch state: r0 the animation, r4 the state, r7 the leader -------------
.org room_update
update:
    push  {r4, r5, r6, lr}
    bl    anim_flags
    mov   r5, r0                        ; the answer the state asked for
    ldr   r0, [r7, #0x81c]
    cmp   r0, #MARIO                    ; only the leader whose model is his
    bne   done
    ldr   r6, state_from_update
update_at:
    add   r6, pc, r6                    ; r6 -> the two bytes
    tst   r5, #0x800                    ; the clip's "another press is taken from here" marker
    movne r0, #1
    strbne r0, [r6, #1]
    ldrb  r0, [r6, #1]
    cmp   r0, #0
    beq   done                          ; not yet
    ldrb  r0, [r6]
    cmp   r0, #LAST_MOVE                ; the last move: nothing follows it
    bcs   done
    mov   r0, r7
    bl    player_of
    cmn   r0, #1
    beq   done                          ; nobody is controlling him
    mov   r4, r0
    b     whose_pad
state_from_update: .word PUNCH_STATE - (update_at + 8)

; ---- has A just been pressed? Read as the walk state reads it: r4 the player ----
.org room_pad
whose_pad:
    ldr   r0, mode_from_here
mode_at:
    ldr   r0, [pc, r0]
    cmp   r0, #0
    beq   pad
    bl    shared_mode
    cmp   r0, #0
    beq   pad
    ldr   r0, game_from_here
game_at:
    ldr   r0, [pc, r0]
    bl    shared_pad
    mov   r4, r0
pad:
    ldr   r1, pads_from_here
pads_at:
    ldr   r1, [pc, r1]
    add   r1, r1, #0x10
    cmp   r4, #4
    addcc r1, r1, r4, lsl #10           ; + player * 0x4f0
    addcc r1, r1, r4, lsl #8
    subcc r1, r1, r4, lsl #4
    ldrb  r0, [r1, #8]                  ; what was pressed this frame
    tst   r0, #1                        ; A
    beq   done
    b     next
mode_from_here: .word mode_object - (mode_at + 8)
game_from_here: .word game_object - (game_at + 8)
pads_from_here: .word pads - (pads_at + 8)

; ---- the next move ----------------------------------------------------------------------------
.org room_next
next:
    ldrb  r1, [r6]
    add   r1, r1, #1
    strh  r1, [r6]                      ; one more move made; no press is being taken
    add   r1, r1, #PUNCH_CLIP           ; its clip
    ldr   r0, [r7, #0x374]              ; the leader's animation
    ldr   r12, [r0]
    mov   r2, #2                        ; on the whole body
    ldr   r12, [r12, #0xc4]
    blx   r12                           ; play it
    bic   r5, r5, #0x80000000           ; and the state does not end on this frame
done:
    mov   r0, r5
    pop   {r4, r5, r6, pc}

; ---- entering the punch state: r0 the animation, r2 which part of the body, r4 the state -------
enter_a:
    ldr   r12, [r4, #0x5c]              ; the leader
    b     enter
enter_b:
    ldr   r12, [r4, #0x28]              ; the same, from the twin's side of the state
enter:
    mov   r1, #0x42                     ; the leaders' punch, as the game had it
    ldr   r12, [r12, #0x81c]
    cmp   r12, #MARIO
    bxne  lr
    ldr   r12, state_from_enter
enter_at:
    add   r12, pc, r12
    mov   r1, #0
    strh  r1, [r12]                     ; no move made yet; no press being taken
    mov   r2, #2                        ; the whole body, moving or not
    mov   r1, #PUNCH_CLIP               ; his first punch
    bx    lr
state_from_enter: .word PUNCH_STATE - (enter_at + 8)

; ---- what a hit reaches and how hard it lands: r7 the leader. Each of these stands where the
; ---- game set one number, and hands back the game's own number for every other leader ----------
.org room_reach
reach:                                  ; r0: the radius searched; s18 (already 4.0): how far ahead of him
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    ldr   r0, radius_theirs
    bxne  lr
    ldr   r0, radius_his
    ldr   r12, ahead_his
    vmov  s18, r12
    bx    lr
furthest:                               ; s16: the furthest from him a hit may land; r0 is the game's, kept
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    ldr   r12, furthest_theirs
    ldreq r12, furthest_his
    vmov  s16, r12
    bx    lr
strength:                               ; r0: the strength of the hit
    ldr   r12, [r7, #0x81c]
    cmp   r12, #MARIO
    ldr   r0, power_theirs
    ldreq r0, power_his
    bx    lr
radius_theirs:   .word RADIUS_THEIRS
radius_his:      .word RADIUS
ahead_his:       .word AHEAD
furthest_theirs: .word FURTHEST_THEIRS
furthest_his:    .word FURTHEST
power_theirs:    .word POWER_THEIRS
power_his:       .word POWER
