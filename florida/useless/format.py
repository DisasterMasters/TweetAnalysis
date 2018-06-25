file = open("Crisislex.txt", "r")
output = open("Crisislex_formatted", "w")
lex = file.read()
lex = lex.split("\n")
output.write(str(lex))
	
