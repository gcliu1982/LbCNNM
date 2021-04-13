function [] = savematfile(spath,obj)
%SAVEMATFILE Summary of this function goes here
%   Detailed explanation goes here
save(spath,'obj','-v7.3');