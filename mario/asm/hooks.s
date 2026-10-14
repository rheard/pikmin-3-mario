; The sixteen words of the game's own code that change. Thirteen become a call to the new code,
; which does what the word did for every other leader; three are a clip's number, in places only
; the leader with Alph's model is sent to. See punch.s, jump.s and idle.s for the far side of each.

; ==== the walk state, where it moves a leader ====
.org 0x60c4fe14                         ; the call to the steering, once a frame
    bl    j_walk

; ==== entering the punch state ====
.org 0x60c55978                         ; "the clip to play is the punch"
    bl    j_enter_a
.org 0x60c559d4                         ; the same in the state's second way in
    bl    j_enter_b

; ==== each frame of the punch state ====
.org 0x60c55a48                         ; "is this the frame it lands?"
    bl    j_flags
.org 0x60c55aa4                         ; the radius of the sphere searched for an enemy, 10.0
    bl    j_reach
.org 0x60c55b14                         ; the furthest away a hit may land, 20.0
    bl    j_furthest
.org 0x60c55c30                         ; the strength of the hit, 3.0
    bl    j_strength
.org 0x60c55c70                         ; the call that hands the enemy the hit
    bl    j_send
.org 0x60c55f28                         ; "has the clip ended?"
    bl    j_update
.org 0x60c55fd0                         ; the same question on the path taken after a kill
    bl    j_update

; ==== someone left standing is given a clip to play once ====
.org 0x609bd0dc                         ; the call that starts it, with the clip just chosen
    bl    i_fidget

; ==== someone left standing rests: the three places a sitting clip is picked, by which leader it is ====
.org 0x609bc9b8                         ; "Alph's sitting clip", as a rest begins
    mov   r0, #SIT_CLIP
.org 0x609bcca8                         ; the same, where a sleeper is sat up again
    mov   r0, #SIT_CLIP
.org 0x609bcda0                         ; the same, where one who stands is sat down
    mov   r0, #SIT_CLIP

; ==== each frame of a rest ====
.org 0x609bcb00                         ; "the clip has finished: was he lying?"
    bl    i_woke
.org 0x609bcd80                         ; "the clip to lie down in is the sleeping clip"
    bl    i_nap
