; Left standing.
;
; Whoever is not being played and has nothing to follow, leader or Pikmin, is given things to do by
; the game: again and again either a clip to play once (a yawn, a chat, a search, a stand or a fret)
; or a rest. A rest plays a sitting clip, chosen by which leader it is, round and round between two
; of its frames. When the rest's wait runs out the sitter is either laid down, in a sleeping clip
; that goes round the same way, or told to finish; a clip told to finish plays on to its end. The
; game then asks what the rest was doing: one who was lying is sat up again, and for one who was
; sitting the rest is over.
;
; In every one of those clips Mario's own bones rest, so he stood stiff and sat in the air. He has
; clips of his own for all of it (IDLE_CLIP and the three after it), and the routines here hand
; them to the game in place of its own, for him and for nobody else. Three more words of the game
; are simply the number of Alph's sitting clip, which becomes his: see hooks.s.

; ---- the game has chosen a clip for someone to play once, and is about to start it: r0 the behaviour,
;      r1 = 1, r2 what is asked for (the clip's number 4 into it), r5 whoever it is ----
.org room_idle
i_fidget:
    ldr   r12, [r5, #0x81c]
    cmp   r12, #MARIO                   ; only the leader whose model is his
    bne   i_asked
    ldr   r12, [r2, #4]
    cmp   r12, #THEIR_STAND             ; the stand every Pikmin shares, in which his bones rest:
    moveq r12, #HIS_STAND               ; ... the leaders' own entry for standing, which plays his stand
    beq   i_put
    ldr   r3, byte_from_idle            ; anything else: the next of the things he does, in turn
i_fidget_at:
    add   r3, pc, r3
    ldrb  r12, [r3]
    add   r12, r12, #1
    cmp   r12, #IDLE_TURNS
    movcs r12, #0
    strb  r12, [r3]
    add   r12, r12, #IDLE_CLIP
i_put:
    str   r12, [r2, #4]
i_asked:
    b     make_once                     ; as the game was about to (it comes back to the game's own code)
byte_from_idle: .word IDLE_STATE - (i_fidget_at + 8)

; ---- a sitter is to be laid down to sleep: r0 its animation, r3 that animation's "play", r4 the rest ----
i_nap:
    ldr   r12, [r4, #0x5c]              ; whoever it is
    ldr   r12, [r12, #0x81c]
    mov   r1, #THEIR_NAP                ; the clip the game lays a sitter down in
    cmp   r12, #MARIO
    moveq r1, #NAP_CLIP                 ; his lying down, sleeping and waking
    bx    lr

; ---- a clip that was told to finish has ended: r0 the rest's flags (kept), r1 what it was doing, r4 the
;      rest. The game asks "was it lying?" (3), and sits such a one up again; else "was it sitting?" (1),
;      and for that one the rest is over. He wakes from lying onto his feet, so for him lying is answered
;      as sitting ----
i_woke:
    cmp   r1, #3
    bxne  lr                            ; not lying: as the game
    ldr   r12, [r4, #0x5c]              ; whoever it is
    ldr   r12, [r12, #0x81c]
    cmp   r12, #MARIO
    moveq r1, #1
    cmp   r1, #3                        ; the game's own test, on what is there now
    bx    lr
