(* ::Section:: *)
(*9 Summary of all checks*)

(* ::Input:: *)
Grid[Prepend[List @@@ $checks, {"check", "passed"}], Frame -> All, Alignment -> Left]
allPassed = And @@ Values[$checks]
