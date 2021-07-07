pro order_mask, ord_mask


; G1xG2 5-8 microns

orders=[14, 15, 16, 17, 18, 19, 20, 21, 22,23]
num_orders = n_elements(orders)
ord_mask = fltarr(256,256)

for i=0,num_orders-1 do begin      ; for each spectral order

  ;map=[[[194,239],[156,202],[124,165],[93,135],[68,106],[42,81],[23,56],[3,36]],$
  ;  [[0,234]  ,[0,255]  ,[0,255] ,[0,255] ,[0,255],[0,255],[0,255],[0,255]]]
  
   ;ord_height=[15,15,15,15,15,15,15,15,15]
   ; Map produced by Rob Lewis on 4/17/2013
   ;map=[[[199,241],[160,206],[126,170],[96,138],[70,108],[46,84],[25,61],[5,40],[1,21]],$
   ;       [[0,232] ,[0,255]  ,[0,255]  ,[0,255] ,[0,255] ,[0,255],[0,255],[0,255],[123,255]]]
   
   ; Map produced by Luke Keller on 4/22/2013
    ord_height=[15,15,15,15,15,15,15,15,15,15]
    map=[[[231,239],[189,238],[150,196],[116,160],[86,128],[60,98],[36,74],[15,51],[0,29],[0,9]],$
          [[0,42],[0,255] ,[0,255]  ,[0,255]  ,[0,255] ,[0,255] ,[0,255],[0,255],[52,255],[184,255]]]
  
   ;ord_height=[11,11,11,11,11,11,11,11,11]

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
;ord_mask = rot(ord_mask,90.0)
atv,ord_mask
;tvscl,ord_mask
;help,ord_mask
;print,moment(ord_mask)
;print,min(ord_mask), max(ord_mask)
;print, ord_mask
writefits,'gxd5-8_order_mask_5_3_2013.fits',ord_mask
end
