import copy 

def merge_sort( arr ,str_pos ,end_pos): 
    if str_pos < end_pos:
        mid = (str_pos + end_pos)//2 
        merge_sort( arr ,str_pos ,mid)
        merge_sort( arr ,mid+1 ,end_pos)
        merge( arr ,str_pos ,mid ,end_pos)
        
def merge( arr ,str_pos ,mid ,end_pos ):    
    arr1 = copy.copy( arr[ str_pos : mid + 1 ] )
    arr2 = copy.copy( arr[ mid + 1 : end_pos + 1 ] )
    
    i ,ind1 ,ind2 = str_pos ,0 ,0
    tot = end_pos - str_pos + 1
    while( ind1 < len(arr1) and ind2 < len(arr2) ):

        while( ind1<len(arr1) and arr1[ind1]<=arr2[ind2] ):
            arr[i] = arr1[ind1]
            i+=1
            ind1+=1
        
        while( ind2<len(arr2) and ( ind1==len(arr1) or arr2[ind2]<arr1[ind1] ) ):
            arr[i] = arr2[ind2]
            i+=1
            ind2+=1
         
    while( ind1 < len(arr1) ):
        arr[i] = arr1[ind1]
        i+=1
        ind1+=1
    
    while( ind2 < len(arr2) ):
        arr[i] = arr2[ind2]
        i+=1
        ind2+=1
    
    del arr1 ,arr2  


        

