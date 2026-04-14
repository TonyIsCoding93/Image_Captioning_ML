


def loadCaptions():

    hashmap = {}
    
    with open("data/archive/captions.txt", "r") as file:
        lines = file.readlines()
        
        for line in lines[1:]:
            seperatedLine = line.strip().split(",", 1)
            if seperatedLine[0] not in hashmap:
                hashmap[seperatedLine[0]] = [seperatedLine[1]]
            else:
                hashmap[seperatedLine[0]].append(seperatedLine[1])
    return hashmap
            


