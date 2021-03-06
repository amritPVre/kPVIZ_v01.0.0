# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:48:42 2022

@author: amrit
"""
import pdfkit
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
from csv import reader
from csv import DictReader





def main():
    # open file in read mode
        with open('data/d2.csv', 'r') as read_obj, open('output.html', 'w') as f_output:
        # pass the file object to DictReader() to get the DictReader object
            csv_dict_reader = DictReader(read_obj)
        # iterate over each line as a ordered dictionary
        # Write the HTML header
            f_output.write("""
                       <html>
<head>
<meta name="generator" content="WYSIWYG Web Builder 17 Trial Version - https://www.wysiwygwebbuilder.com">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="font-awesome.min.css" rel="stylesheet">
<link href="flexible_layout.css" rel="stylesheet">
<link href="index.css" rel="stylesheet">
<script src="jquery-1.12.4.min.js"></script>
<script src="wb.canvas.min.js"></script>
<link rel="stylesheet" href="magnificpopup/magnific-popup.css">
<script src="magnificpopup/jquery.magnific-popup.min.js"></script>
<script src="index.js"></script>
</head>
                        """)
            for row in csv_dict_reader:
                ## row variable is a dictionary that represents a row in csv
                f_output.write(f"""
                           
                           
                           <body>
   <div id="container">
      <a href="https://www.wysiwygwebbuilder.com" target="_blank"><img src="" alt="WYSIWYG Web Builder" style="position:absolute;left:442px;top:212px;margin:0;border-width:0;z-index:250" width="1" height="1"></a>
   </div>
   <div id="wb_header">
      <div id="header">
         <div class="row">
            <div class="col-1">
               <div id="wb_Image1" style="display:inline-block;width:100%;height:auto;z-index:0;">
                  <img src="images/k-removebg-preview.png" id="Image1" alt="" width="178" height="70">
               </div>
            </div>
            <div class="col-2">
               <div id="wb_Heading1" style="display:inline-block;width:100%;z-index:1;">
                  <h1 id="Heading1">Techno-Commercial Summary of Solar PV Plant</h1>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid13">
      <div id="LayoutGrid13">
         <div class="row">
            <div class="col-1">
               <div id="wb_Heading2" style="display:inline-block;width:100%;z-index:2;">
                  <h4 id="Heading2">{row['Company_Name']}  |  Presented by {row['Company_representative']}</h4>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid4">
      <div id="LayoutGrid4">
         <div class="row">
            <div class="col-1">
               <hr id="Line1" style="display:block;width:100%;z-index:3;">
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid5">
      <div id="LayoutGrid5">
         <div class="row">
            <div class="col-1">
               <div id="wb_Shape3" style="display:inline-block;width:280px;height:81px;z-index:4;position:relative;">
                  <canvas id="Shape3" height="81" width="280"></canvas><div id="Shape3_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>DC Capacity:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['DC_Cap']}</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"> </span><span style="color:#F2F2F2;font-family:Arial;font-size:21px;">kWp</span></div></div>
               </div>
            </div>
            <div class="col-2">
               <div id="wb_Shape2" style="display:inline-block;width:280px;height:81px;z-index:5;position:relative;">
                  <canvas id="Shape2" height="81" width="280"></canvas><div id="Shape2_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>Annual Energy Yield:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['E_Grid']}</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"> </span><span style="color:#F2F2F2;font-family:Arial;font-size:21px;">MWh</span></div></div>
               </div>
            </div>
            <div class="col-3">
               <div id="wb_Shape1" style="display:inline-block;width:280px;height:81px;z-index:6;position:relative;">
                  <canvas id="Shape1" height="81" width="280"></canvas><div id="Shape1_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>Annual PR:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['PR']}</strong></span></div></div>
               </div>
            </div>
            <div class="col-4">
               <div id="wb_Shape4" style="display:inline-block;width:280px;height:81px;z-index:7;position:relative;">
                  <canvas id="Shape4" height="81" width="280"></canvas><div id="Shape4_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>AC Capacity:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['AC_Cap']}</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"> </span><span style="color:#F2F2F2;font-family:Arial;font-size:21px;">kW</span></div></div>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid7">
      <div id="LayoutGrid7">
         <div class="row">
            <div class="col-1">
               <div id="wb_Shape5" style="display:inline-block;width:329px;height:81px;z-index:8;position:relative;">
                  <canvas id="Shape5" height="81" width="329"></canvas><div id="Shape5_text"><div><span style="color:#FFFFFF;font-family:Arial;font-size:21px;"><strong>Gross Solar Income in 25 Years:</strong></span><span style="color:#FFFFFF;font-family:Arial;font-size:17px;"><br></span><span style="color:#FFFFFF;font-family:Arial;font-size:29px;"><strong>${row['Gross_25']}M</strong></span></div></div>
               </div>
            </div>
            <div class="col-2">
               <div id="wb_Shape7" style="display:inline-block;width:329px;height:81px;z-index:9;position:relative;">
                  <canvas id="Shape7" height="81" width="329"></canvas><div id="Shape7_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>Specific Yield:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['Spec_E']}</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"> </span><span style="color:#F2F2F2;font-family:Arial;font-size:21px;">kWh/kWp/Year</span></div></div>
               </div>
            </div>
            <div class="col-3">
               <div id="wb_Shape6" style="display:inline-block;width:329px;height:81px;z-index:10;position:relative;">
                  <canvas id="Shape6" height="81" width="329"></canvas><div id="Shape6_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:21px;"><strong>Upfront Project Cost:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>${row['Proj_Cost']}M</strong></span></div></div>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_galleryGrid">
      <div id="galleryGrid">
         <div class="row">
            <div class="col-1">
               <div id="wb_gallery" style="display:inline-block;width:100%;z-index:19;">
                  <div id="gallery">
                     <div class="thumbnails">
                        <figure class="thumbnail">
                           <div class="frame">
                              <a href="Monthly_E_Grid.png" data-rel="" title="Monthly Energy Yield Data"><img alt="" src="Monthly_E_Grid.png" class="image"><figcaption class="caption"><span class="title">Monthly Energy Yield Data</span></figcaption></a>
                           </div>
                        </figure>
                        <figure class="thumbnail">
                           <div class="frame">
                              <a href="Monthly_GHI_GII.png" data-rel="" title="Monthly GHI and GII Data"><img alt="" src="Monthly_GHI_GII.png" class="image"><figcaption class="caption"><span class="title">Monthly GHI and GII Data</span></figcaption></a>
                           </div>
                        </figure>
                        <div class="clearfix visible-col2"></div>
                     </div>
                  </div>
               </div>
               <div id="wb_LayoutGrid9">
                  <div id="LayoutGrid9">
                     <div class="row">
                        <div class="col-1">
                           <div id="wb_Text9">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>Monthly Solar Energy Yield Data</strong></span>
                           </div>
                        </div>
                        <div class="col-2">
                           <div id="wb_Text10">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>Monthly GHI &amp; GII Data</strong></span>
                           </div>
                        </div>
                     </div>
                  </div>
               </div>
               <div id="wb_LayoutGrid8">
                  <div id="LayoutGrid8">
                     <div class="row">
                        <div class="col-1">
                           <div id="wb_Text7">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>{row['Electricity_Tariff']}: </strong>${row['E_tariff']}/kWh</span>
                           </div>
                        </div>
                        <div class="col-2">
                           <div id="wb_Text8">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>Yearly O&amp;M Cost: </strong>{row['OnM']}% of Total Project Cost</span>
                           </div>
                        </div>
                     </div>
                  </div>
               </div>
               <div id="wb_LayoutGrid10">
                  <div id="LayoutGrid10">
                     <div class="row">
                        <div class="col-1">
                           <div id="wb_Text11">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>Year-On-Year Escalation: </strong>{row['Tariff_esc']}%</span>
                           </div>
                        </div>
                        <div class="col-2">
                           <div id="wb_Text12">
                              <span style="color:#000000;font-family:Arial;font-size:21px;"><strong>Year-On-Year Escalation: </strong>{row['OnM_esc']}%</span>
                           </div>
                        </div>
                     </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid11">
      <div id="LayoutGrid11">
         <div class="row">
            <div class="col-1">
               <div id="wb_Shape8" style="display:inline-block;width:329px;height:81px;z-index:24;position:relative;">
                  <canvas id="Shape8" height="81" width="329"></canvas><div id="Shape8_text"><div><span style="color:#FFFFFF;font-family:Arial;font-size:27px;"><strong>LCOE:</strong></span><span style="color:#FFFFFF;font-family:Arial;font-size:17px;"><br></span><span style="color:#FFFFFF;font-family:Arial;font-size:29px;"><strong>${row['LCOE']}/kWh</strong></span></div></div>
               </div>
            </div>
            <div class="col-2">
               <div id="wb_Shape10" style="display:inline-block;width:329px;height:81px;z-index:25;position:relative;">
                  <canvas id="Shape10" height="81" width="329"></canvas><div id="Shape10_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:27px;"><strong>Payback Period:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['PBP']} Years</strong></span></div></div>
               </div>
            </div>
            <div class="col-3">
               <div id="wb_Shape9" style="display:inline-block;width:329px;height:81px;z-index:26;position:relative;">
                  <canvas id="Shape9" height="81" width="329"></canvas><div id="Shape9_text"><div><span style="color:#F2F2F2;font-family:Arial;font-size:27px;"><strong>Internal Rate of Return:</strong></span><span style="color:#F2F2F2;font-family:Arial;font-size:17px;"><br></span><span style="color:#F2F2F2;font-family:Arial;font-size:29px;"><strong>{row['IRR']} %</strong></span></div></div>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid12">
      <div id="LayoutGrid12">
         <div class="row">
            <div class="col-1">
            </div>
            <div class="col-2">
               <picture id="wb_Picture1" style="display:block;width:100%;height:auto;z-index:27">
                  <img src="Cashflow_25.png" id="Picture1" alt="" title="Cashflow Projection">
               </picture>
            </div>
            <div class="col-3">
            </div>
            <div class="col-4">
            </div>
            <div class="col-5">
            </div>
         </div>
      </div>
   </div>
   <div id="wb_Divider1">
      <div id="Divider1-overlay"></div>
      <div id="Divider1">
         <div class="row">
            <div class="col-1">
               <hr id="DividerLine1" style="display:block;width:100%;z-index:28;">
            </div>
         </div>
      </div>
   </div>
   <div id="wb_count-up">
      <div id="count-up">
         <div class="row">
            <div class="col-1">
               <div id="wb_FontAwesomeIcon1" style="display:inline-block;width:64px;height:64px;text-align:center;z-index:29;">
                  <div id="FontAwesomeIcon1"><i class="FontAwesomeIcon1"></i></div>
               </div>
               <div id="wb_Heading3" style="display:inline-block;width:100%;z-index:30;">
                  <h1 id="Heading3">{row['CO2']}</h1>
               </div>
               <div id="wb_Text1">
                  <span style="color:#000000;font-family:Arial;font-size:16px;">Tonnes of CO2 Offset per Year</span>
               </div>
            </div>
            <div class="col-2">
               <div id="wb_FontAwesomeIcon2" style="display:inline-block;width:64px;height:64px;text-align:center;z-index:32;">
                  <div id="FontAwesomeIcon2"><i class="FontAwesomeIcon2"></i></div>
               </div>
               <div id="wb_Heading4" style="display:inline-block;width:100%;z-index:33;">
                  <h1 id="Heading4">{row['Cars']}</h1>
               </div>
               <div id="wb_Text2">
                  <span style="color:#000000;font-family:Arial;font-size:16px;">Cars Taken off from the Road</span>
               </div>
            </div>
            <div class="col-3">
               <div id="wb_IconFont1" style="display:inline-block;width:100%;text-align:center;z-index:35;">
                  <div id="IconFont1"><i class="IconFont1"></i></div>
               </div>
               <div id="wb_Heading5" style="display:inline-block;width:100%;z-index:36;">
                  <h1 id="Heading5">{row['Houses']}</h1>
               </div>
               <div id="wb_Text3">
                  <span style="color:#000000;font-family:Arial;font-size:16px;">Houses Energy Met per Year</span>
               </div>
            </div>
            <div class="col-4">
               <div id="wb_IconFont2" style="display:inline-block;width:100%;text-align:center;z-index:38;">
                  <div id="IconFont2"><i class="fa fa-tree"></i></div>
               </div>
               <div id="wb_Heading6" style="display:inline-block;width:100%;z-index:39;">
                  <h1 id="Heading6">{row['Trees']}</h1>
               </div>
               <div id="wb_Text4">
                  <span style="color:#000000;font-family:Arial;font-size:16px;">Tree Seedlings grown in 10 Years</span>
               </div>
            </div>
         </div>
      </div>
   </div>
   <div id="wb_LayoutGrid6">
      <div id="LayoutGrid6">
         <div class="row">
            <div class="col-1">
               <div id="wb_Text6">
                  <span style="color:#FFFFFF;">Developed By Amrit Mandal</span>
               </div>
            </div>
         </div>
      </div>
   </div>
                           """)
            f_output.write("</body>\n</html>\n")
            
if __name__ == '__pdf_creator__':
    main()       
            