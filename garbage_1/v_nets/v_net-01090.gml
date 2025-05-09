graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1090
  arrival_time 22775.81192886632
  lifetime 790.5223380130418
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 43
    gpu 6
    rom 29
  ]
  node [
    id 1
    label "1"
    cpu 20
    gpu 28
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 21
    gpu 21
    rom 20
  ]
  node [
    id 3
    label "3"
    cpu 21
    gpu 31
    rom 5
  ]
  node [
    id 4
    label "4"
    cpu 7
    gpu 3
    rom 7
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 43
    rom 17
  ]
  node [
    id 6
    label "6"
    cpu 36
    gpu 21
    rom 14
  ]
  node [
    id 7
    label "7"
    cpu 31
    gpu 16
    rom 35
  ]
  node [
    id 8
    label "8"
    cpu 2
    gpu 40
    rom 39
  ]
  node [
    id 9
    label "9"
    cpu 6
    gpu 31
    rom 18
  ]
  node [
    id 10
    label "10"
    cpu 31
    gpu 40
    rom 3
  ]
  node [
    id 11
    label "11"
    cpu 36
    gpu 18
    rom 0
  ]
  node [
    id 12
    label "12"
    cpu 45
    gpu 6
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 22
  ]
  edge [
    source 1
    target 2
    bw 33
  ]
  edge [
    source 2
    target 3
    bw 44
  ]
  edge [
    source 3
    target 4
    bw 5
  ]
  edge [
    source 4
    target 5
    bw 27
  ]
  edge [
    source 5
    target 6
    bw 40
  ]
  edge [
    source 6
    target 7
    bw 7
  ]
  edge [
    source 7
    target 8
    bw 28
  ]
  edge [
    source 8
    target 9
    bw 5
  ]
  edge [
    source 9
    target 10
    bw 0
  ]
  edge [
    source 10
    target 11
    bw 5
  ]
  edge [
    source 11
    target 12
    bw 12
  ]
]
