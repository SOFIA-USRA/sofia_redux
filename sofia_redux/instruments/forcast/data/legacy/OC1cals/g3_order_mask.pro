pro g3_order_mask, ord_mask


; G4xG3 8 - 13 microns

orders=[8, 9, 10, 11, 12]
num_orders = n_elements(orders)
ord_mask = fltarr(256,256)

for i=0,num_orders-1 do begin      ; for each spectral order


    map=[[[196,236],[127,192],[72,132],[28, 84],[0, 45]], $
            [[0,138],[0,255],[0,255],[0,255],[59,255]]] 


        	
    ord_height = [23, 23, 23, 23, 23]

    slope= float(map[1,i,0]-map[0,i,0])/float(map[1,i,1]-map[0,i,1])

    xvalue=findgen(map[1,i,1]-map[0,i,1]+1)
  
    ;print,'xvalue',xvalue
    ;yvalue, the spatial direction positions of the order in the array,
    ; calculated from the order slope
    
    yvalue=slope*xvalue+map[0,i,0]
    
    if map[0,i,1] ne 0 then begin
       
       for j = 0,n_elements(xvalue)-1 do begin
    xvalue[j] = xvalue[j] + map[0,i,1]  
    ;yvalue[j] = slope*xvalue[j]-(map[1,i,0]-map[0,i,0])
    ;print,slope*xvalue[j]-(map[1,i,0]-map[0,i,0])
      endfor

       for n = 0,n_elements(xvalue)-1 do begin
       b = map[1,i,0]-slope*xvalue[j-1]
       ;print,b
       yvalue[n] = slope*xvalue[n] + b
       endfor


    endif 
    dy = (ord_height)[i]
    
    for j=0,ord_height[i]-1 do begin

        ord_mask(xvalue,yvalue+j) = 1.0

    endfor
    ;ord_mask(where(ord_mask(xvalue,yvalue:yvalue+dy))) = 0.0
 
endfor
ord_mask = rot(ord_mask,-90.0)
atv,ord_mask
;tvscl,ord_mask
;help,ord_mask
;print,moment(ord_mask)
;print,min(ord_mask), max(ord_mask)
;print, ord_mask
writefits,'gxd8-13_order_mask_5_7_2013.fits',ord_mask
end
