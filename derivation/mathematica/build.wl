(* Generate doped_gamma.nb from the sources:  wolframscript -file build.wl nb=doped_gamma.nb src/*.wl

   Each source file is a sequence of blocks, each starting with a line (* ::Style:: *). Input blocks are
   evaluated in order (each expression limited to 10 s; slower ones are reported), and the notebook gets
   their code as input cells, with the non-Null results as output cells; all other blocks become text
   cells of that style. Exits with status 1 if a check fails. *)
args = Rest[$ScriptCommandLine];
nbFile = SelectFirst[args, StringStartsQ[#, "nb="] &, None];
files = Select[args, ! StringStartsQ[#, "nb="] &];

blocks[file_] := Module[{out = {}, style = None, buf = {}},
  Do[If[StringMatchQ[l, "(* ::" ~~ __ ~~ ":: *)"],
     If[style =!= None, AppendTo[out, {style, buf}]]; style = StringTake[l, {6, -6}]; buf = {},
     AppendTo[buf, l]],
   {l, ReadList[file, String, NullRecords -> True]}];
  If[style =!= None, AppendTo[out, {style, buf}]];
  out];

short[e_] := StringTake[ToString[e, InputForm], UpTo[150]];
evaluate[e_] := Module[{t, r},
  {t, r} = AbsoluteTiming[TimeConstrained[ReleaseHold[e], 10, $TimedOut]];
  Which[r === $TimedOut, Print["TIMEOUT: ", short[e]], t > 2, Print["SLOW (", Round[t, 0.1], " s): ", short[e]]];
  r];

cells = {};
Do[Module[{style = b[[1]], lines = DeleteCases[b[[2]], ""]},
   If[style === "Input",
    Module[{code = StringRiffle[lines, "\n"]},
     AppendTo[cells, Cell[code, "Input"]];
     Do[With[{r = evaluate[e]}, If[r =!= Null, AppendTo[cells, Cell[BoxData[ToBoxes[r, StandardForm]], "Output"]]]],
      {e, ReadList[StringToStream[code], Hold[Expression]]}]],
    AppendTo[cells, Cell[StringRiffle[StringReplace[lines, StartOfString ~~ "(*" ~~ s___ ~~ "*)" ~~ EndOfString :> s], "\n"], style]]]],
  {f, files}, {b, blocks[f]}];

failed = Keys[Select[$checks, ! TrueQ[Last[#]] &]];
Print[Length[$checks], " checks, ", Length[failed], " failed", If[failed === {}, "", ": " <> StringRiffle[failed, ", "]]];

(* the front end parses the input cells into boxes, so that the notebook shows formatted code *)
If[nbFile =!= None,
  cells = UsingFrontEnd[cells /. Cell[code_String, "Input"] :>
       Cell[First[FrontEndExecute[FrontEnd`UndocumentedTestFEParserPacket[code, False]]], "Input"]];
  Put[Notebook[cells, StyleDefinitions -> "Default.nb"], StringDrop[nbFile, 3]]];
Exit[If[failed === {}, 0, 1]];
